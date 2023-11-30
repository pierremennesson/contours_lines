import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from itertools import combinations
from shapely.geometry import LineString,Polygon
from shapely.wkt import loads
import time

        
#OSM GRAPH CORRECTION

def to_multi_graph(G_osm):
    """This function turn the osm multidigraph to a multigraph.
    One cannot just call nx.MultiGraph() because the edge data depends on the orientation.

    Parameters
    ----------
    G_osm : an osm multidigraph


    Returns
    -------
    the induced multigraph

    """
    new_edges=[]
    for u,v,k in G_osm.edges(keys=True):
        if not( (v,u,k) in new_edges):
            new_edges.append((u,v,k))
    return nx.edge_subgraph(G_osm,new_edges)




def add_missing_geometries(G_osm,max_line_length=float('inf')):
    """This function adds a geometry to the osmn edges when it's missing
    by adding a straight line.


    Parameters
    ----------
    G_osm : an osm multigraph

    max_line_length : the maximum distance between two points for a line to be added


    """
    attr,to_be_removed={},[]
    for u,v,k,d in G_osm.edges(data=True,keys=True):
        removed=False
        if not('geometry' in d):
            if 'length' in d:
                if d['length']<max_line_length:
                    ls=LineString([[G_osm.nodes()[u]['x'],G_osm.nodes()[u]['y']],[G_osm.nodes()[v]['x'],G_osm.nodes()[v]['y']]])
                    d.update({'geometry':ls})
                    attr[(u,v,k)]=d
                else:
                    removed=True
            else:
                removed=True
        if removed:
            to_be_removed.append((u,v,k))
    G_osm.remove_edges_from(to_be_removed)
    nx.set_edge_attributes(G_osm,attr)

        
#ADD TO DATABASE

#assume geometry is in some utm crs
def add_contour_lines_to_database(df,cursor,source=None,is_closed=None,contours_lines_table_name="contours_lines"):
    if is_closed is not None:
        df['is_closed']=is_closed
    else:
        df['is_closed']=df['geometry'].apply(lambda ls:ls.is_closed)
    df['length']=df['geometry'].apply(lambda ls:ls.length)
    df['number_points']=df['geometry'].apply(lambda ls:len(ls.coords))
    df=df.to_crs('epsg:4326')
    if source is None:
        string_list=',\n'.join(['(%s,%f,%f,%f,ST_GeomFromText(\'%s\'))'%(str(row['is_closed']).upper(),row['elevation'],row['length'],row['number_points'],row['geometry'].wkt)  for _,row in df.iterrows()])
        cmd="""INSERT INTO %s (`is_closed`,`elevation`,`length`,`number_points`,`geometry`) VALUES %s;"""%(contours_lines_table_name,string_list)
    else:
        string_list=',\n'.join(['(%i,%s,%f,%f,%f,ST_GeomFromText(\'%s\'))'%(source,str(row['is_closed']).upper(),row['elevation'],row['length'],row['number_points'],row['geometry'].wkt)  for _,row in df.iterrows()])
        cmd="""INSERT INTO %s (`source`,`is_closed`,`elevation`,`length`,`number_points`,`geometry`) VALUES %s;"""%(contours_lines_table_name,string_list)
    cursor.execute(cmd)
        

        

#BUILD GRAPH

def build_graph(cursor,contours_lines_table_name="contours_lines",tree_edges_table_name="tree_edges",with_distance=None):
    G=nx.DiGraph()
    if with_distance is not None:
        if with_distance:
            cmd="SELECT * FROM %s WHERE distance IS NOT NULL"%tree_edges_table_name
        else:
            cmd="SELECT * FROM %s WHERE distance IS NULL"%tree_edges_table_name
    else:
        cmd="SELECT * FROM %s"%tree_edges_table_name
    cursor.execute(cmd)
    if with_distance is not None and with_distance:
        G.add_edges_from([(elem['begin'],elem['end'],{'distance':elem['distance']}) for elem in cursor.fetchall()])
    else:
        G.add_edges_from([(elem['begin'],elem['end']) for elem in cursor.fetchall()])

    G.add_node('root')
    for node,deg in G.in_degree():
        if node!='root' and deg==0:
            G.add_edge('root',node)

    cmd="SELECT id,number_points,elevation FROM %s"%contours_lines_table_name
    cursor.execute(cmd)
    nx.set_node_attributes(G,{elem['id']:{'number_points':elem['number_points'],'elevation':elem['elevation']} for elem in cursor.fetchall()})
    nx.set_node_attributes(G,{'root':{'number_points':0}})
    add_depth(G)
    G.remove_node('root')
    return G

def add_depth(G):
    paths=nx.shortest_path_length(G,source='root')
    nx.set_node_attributes(G,{node:{'depth':length} for node,length in paths.items()})






#ACCESS DATA

def get_level_contours_df(cursor,level,contours_lines_table_name="contours_lines",is_closed=True):
    cmd="SELECT id,is_closed,elevation,ST_asText(geometry) AS geometry FROM %s WHERE elevation=%f" %(contours_lines_table_name,level)
    if is_closed is not None:
        if is_closed:
            cmd=cmd+" AND is_closed"
        else:
            cmd=cmd+" AND NOT is_closed"
    cursor.execute(cmd)
    level_open_contours_df=pd.DataFrame(cursor.fetchall())
    if len(level_open_contours_df)==0:
        return None
    level_open_contours_df['geometry']=level_open_contours_df['geometry'].apply(lambda ls:loads(ls))
    level_open_contours_df=level_open_contours_df.set_index('id',drop=False)
    level_open_contours_df=gpd.GeoDataFrame(level_open_contours_df,geometry='geometry',crs='epsg:4326')
    return level_open_contours_df

def get_nodes_data(cursor,nodes,contours_lines_table_name="contours_lines"):
    nodes_list='(%s)'%','.join([str(node) for node in nodes if node!='root'])
    cmd="SELECT id,elevation,number_points,ST_asText(geometry) AS geometry FROM %s WHERE id IN %s" %(contours_lines_table_name,nodes_list)
    cursor.execute(cmd)
    data= cursor.fetchall()
    data=pd.DataFrame(data)
    data['geometry']=loads(data['geometry'])
    data=gpd.GeoDataFrame(data,geometry='geometry',crs='epsg:4326')

    return data

#DEPTH SEARCH

def decreasing_depth_intersections(osm_edges_df,osm_crs,G,cursor,max_depth,min_depth=0,depth_step=10,max_delta_time_per_level=None,max_delta_time_total=600.):
    t1=time.time()
    max_delta_time_per_step=None
    if max_delta_time_per_level is not None:
        max_delta_time_per_step=max_delta_time_per_level*depth_step
    intersection,current_osm_edges=None,None
    for depth in range(max_depth,min_depth,-depth_step):
        t2=time.time()
        contour_nodes=[node for node,data in G.nodes(data=True) if depth-depth_step<data['depth']<=depth]
        contours_df=get_nodes_data(cursor,contour_nodes)
        contours_df['depth']=contours_df['id'].apply(lambda node:G.nodes()[node]['depth'])
        contours_df=contours_df.to_crs(osm_crs)
        local_intersection=gpd.overlay(contours_df,osm_edges_df,keep_geom_type=False).explode(index_parts=False)
        if len(local_intersection)>0:
            local_intersection['edge_coordinate']=local_intersection.apply(lambda row:osm_edges_df.loc[row['id_edge']]['geometry'].project(row['geometry']),axis=1)
        if intersection is None:
            intersection=local_intersection
            current_osm_edges=set(local_intersection['edge'])
        else:
            intersection=pd.concat([intersection,local_intersection],ignore_index=True)
            current_osm_edges=current_osm_edges.union(set(local_intersection['edge']))

        terminated_edges=current_osm_edges.difference(local_intersection[local_intersection.depth==depth-depth_step+1]['edge'])
        if len(terminated_edges)>0:
            osm_edges_df=osm_edges_df[~osm_edges_df['edge'].apply(lambda edge: edge in terminated_edges)]
        t3=time.time()
        print('%i<depth<=%i :%f '%(depth-depth_step,depth,t3-t2))
        if max_delta_time_total is not None and (t3-t1)>max_delta_time_total:
            break
        if max_delta_time_per_step is not None and (t3-t2)>max_delta_time_per_step:
            break
    return intersection,depth-depth_step+1

#MERGE LINES

def build_open_contour_graph(level_open_contours_df,max_distance=1000,coeff_restriction=0.1,MLS=None):
    G_extremities=nx.Graph()
    for index,row in level_open_contours_df.iterrows():
        ls=row['geometry']
        length=ls.length
        pt1,pt2=ls.boundary.geoms
        G_extremities.add_node((index,0),geometry=pt1,x=pt1.x,y=pt1.y,length=length)
        G_extremities.add_node((index,1),geometry=pt2,x=pt2.x,y=pt2.y,length=length)
        G_extremities.add_edge((index,0),(index,1),geometry=ls,edge_type='contour',distance=length)


    for (node_1,data_1),(node_2,data_2) in combinations(G_extremities.nodes(data=True),2):
        if node_1[0]!=node_2[0]:
            ls=LineString([data_1['geometry'],data_2['geometry']])
            distance=ls.length
            pt1=ls.interpolate(coeff_restriction*distance)
            pt2=ls.interpolate((1-coeff_restriction)*distance)
            ls=LineString([pt1,pt2])
            distance=ls.length
            if distance<max_distance:
                if MLS is not None:
                    if not(ls.intersects(MLS)):
                        G_extremities.add_edge(node_1,node_2,geometry=ls,edge_type='join',distance=distance)
                else:
                    G_extremities.add_edge(node_1,node_2,geometry=ls,edge_type='join',distance=distance)

    return G_extremities

    

def clean_graph(G_extremities):
    bad_edges,edges=[],G_extremities.edges(data=True)
    geoms_join=gpd.GeoDataFrame([{'edge':(u,v),'first':u,'end':v,'geometry':d['geometry'],'distance':d['distance']} for u,v,d in edges if d['edge_type']=='join'])
    intersection=gpd.overlay(geoms_join,geoms_join,keep_geom_type=False)
    intersection=intersection[intersection.edge_1!=intersection.edge_2]
    while(len(intersection)>0):
        argmax=np.argmax(intersection['distance_1'])
        edge=intersection.iloc[argmax]['edge_1']
        intersection=intersection[(intersection.edge_1!=edge)&(intersection.edge_2!=edge)]
        bad_edges.append(edge)
    G_extremities.remove_edges_from(bad_edges)


def build_contour_lines_from_extremities_graph(G_extremities):
    G_contour_lines=nx.DiGraph()
    for (node_1,index_1),(node_2,index_2) in G_extremities.edges():
        if node_1!=node_2:
            if index_1==0:
                G_contour_lines.add_edge(node_1,node_2)
            else:
                G_contour_lines.add_edge(node_2,node_1)
    return G_contour_lines
    

def flip(node):
    return (node[0],1-node[1])


def is_a_good_candidate(G_deleted,neighbor,start_node):
    next_node=flip(neighbor)
    data=G_deleted.get_edge_data(neighbor,next_node)
    G_deleted.remove_edge(neighbor,next_node)
    res=nx.has_path(G_deleted,next_node,start_node)
    G_deleted.add_edge(neighbor,next_node,**data)
    return res

def get_walk(G,start_node):
    G_deleted=deepcopy(G).copy()
    path=[start_node,flip(start_node)]
    G_deleted.remove_node(path[-1])

    while(True):
        last_node=path[-1]
        neighbors=[neighbor for neighbor in G.neighbors(last_node) if not(neighbor in path) and is_a_good_candidate(G_deleted,neighbor,start_node)]
        if len(neighbors)==0:
            break
        else:
            neighbors=sorted(neighbors,key=lambda neigh:G.get_edge_data(last_node,neigh)['distance'])
            path.append(neighbors[0])
            path.append(flip(neighbors[0]))
            G_deleted.remove_node(path[-2])
            G_deleted.remove_node(path[-1])
    success=start_node in G.neighbors(path[-1])
    G_deleted.remove_node(path[0])
    return path,G_deleted,success



def cycle_decomposition(G_extremities):
    cycles,failed_cycles=[],[]
    while(len(G_extremities.nodes())>0):
        bridges=list(nx.bridges(G_extremities))
        starting_edges=[edge for edge in G_extremities.edges() if not(edge in bridges or (edge[1],edge[0]) in bridges)]
        starting_edges=sorted(starting_edges,key=lambda edge:G_extremities.get_edge_data(*edge)['distance'])
        if len(starting_edges)==0:
            break
        start_node=starting_edges[-1][0]
        walk,G_deleted,success=get_walk(G_extremities,start_node)
        if success:
            walk.append(walk[0])
            cycles.append(walk)
        else:
            failed_cycles.append(walk)
        G_extremities=G_deleted
    return cycles,failed_cycles
                

def get_contour_lines_from_elevation_df(level_open_contours_df,max_distance=10,MLS=None):
    G_extremities=build_open_contour_graph(level_open_contours_df,max_distance=max_distance,MLS=MLS)
    clean_graph(G_extremities)
    cycles,failed_walks=cycle_decomposition(G_extremities)
    contour_lines=[]
    for cycle in cycles:
        contour_line_coords=[]
        for k in range(len(cycle)-1):
            node_1,node_2=cycle[k],cycle[k+1]
            ls=G_extremities.get_edge_data(node_1,node_2)['geometry']
            coords=list(ls.coords)
            if node_1[0]==node_2[0] and node_1[1]==1:
                coords=coords[::-1]
            contour_line_coords+=coords
        contour_lines.append(LineString(contour_line_coords))
    return contour_lines

    


