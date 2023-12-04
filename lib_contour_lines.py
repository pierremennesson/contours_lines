import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from itertools import combinations
from shapely.geometry import LineString,Polygon
from shapely.wkt import loads
import time
from scipy.optimize import minimize
from lib_merging import get_contour_lines_from_elevation_df

        
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
def add_contour_lines_to_database_from_df(df,cursor,source=None,is_closed=None,contours_lines_table_name="contours_lines"):
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
        

def add_contour_lines_to_database(file_paths,cursor,contours_lines_table_name="contours_lines"):
    for k,file_path in enumerate(file_paths[:]):
        print(file_path)
        df=gpd.read_file(file_path) 
        df['elevation']=df['ALTITUDE']
        add_contour_lines_to_database_from_df(df,cursor,source=k,contours_lines_table_name=contours_lines_table_name)
        t3=time.time()
        cmd="""SELECT c1.id AS id_1,ST_asText(c1.geometry) AS geometry_1,c2.id AS id_2,ST_asText(c2.geometry) AS geometry_2 FROM
        (SELECT * FROM %s WHERE source!=%i) AS c1
        JOIN (SELECT * FROM %s WHERE source=%i) AS c2
        ON c1.elevation=c2.elevation AND c1.length=c2.length
        
        ;
        
        """%(contours_lines_table_name,k,contours_lines_table_name,k)
        cursor.execute(cmd)
        L=cursor.fetchall()
        t4=time.time()
        print('getting pairs took %f'%(t4-t3))
        bad_ids=[elem['id_2'] for elem in L if loads(elem['geometry_1'])==loads(elem['geometry_2'])]

        t5=time.time()
        print('comparing pairs took %f'%(t5-t4))
        if len(bad_ids)>0:
            bad_ids_string='(%s)'%', '.join([str(bad_id) for bad_id in bad_ids])
            cmd="DELETE FROM %s WHERE id IN %s"%(contours_lines_table_name,bad_ids_string)
            cursor.execute(cmd)
            t6=time.time()
            print('deleting took %f'%(t6-t5))

        
#MERGING

def add_merged_contours_lines(cursor,contours_lines_table_name="contours_lines"):
    cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%contours_lines_table_name
    cmd=cursor.execute(cmd)
    elevations=np.array([elem['elevation'] for elem in cursor.fetchall()])
    for level in elevations:
        print(level)
        level_open_contours_df=get_level_contours_df(cursor,level,contours_lines_table_name,is_closed=False)
        if level_open_contours_df is not None:
            crs=level_open_contours_df.estimate_utm_crs()
            level_open_contours_df=level_open_contours_df.to_crs(crs)
            merged_contours_lines,merged_nodes=get_contour_lines_from_elevation_df(level_open_contours_df)
            if len(merged_contours_lines)>0:
                merged_contours_df=gpd.GeoDataFrame([{'geometry':ls} for ls in merged_contours_lines],geometry='geometry',crs=crs)
                merged_contours_df['elevation']=level
                add_contour_lines_to_database_from_df(merged_contours_df,cursor,is_closed=True)

                cursor.execute(cmd)
            # if len(merged_nodes)>0:
            #     merge_nodes_string='(%s)'%', '.join([str(merge_node) for merge_node in merged_nodes])
            #     cmd="DELETE FROM %s WHERE id IN %s"%(contours_lines_table_name,merge_nodes_string)

#BUILD GRAPH

def build_graph(cursor,contours_lines_table_name="contours_lines",contours_edges_table_name="contours_edges",with_distance=None):
    G=nx.DiGraph()
    if with_distance is not None:
        if with_distance:
            cmd="SELECT * FROM %s WHERE distance IS NOT NULL"%contours_edges_table_name
        else:
            cmd="SELECT * FROM %s WHERE distance IS NULL"%contours_edges_table_name
    else:
        cmd="SELECT * FROM %s"%contours_edges_table_name
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

#INTERSECTION


def increasing_elevations_intersections(osm_edges_df,osm_crs,cursor,contours_lines_table_name="contours_lines",elevation_step=10,elevation_cut=10.):
    t1=time.time()
    cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%contours_lines_table_name
    cmd=cursor.execute(cmd)
    elevations=np.array([elem['elevation'] for elem in cursor.fetchall()])

    intersection,current_osm_edges=None,None
    for i in range(0,len(elevations)-elevation_step,elevation_step):
        t2=time.time()
        min_elevation,max_elevation=elevations[i],elevations[i+elevation_step]
        cmd="SELECT id FROM %s WHERE elevation>=%f AND elevation<%i"%(contours_lines_table_name,min_elevation,max_elevation)
        cursor.execute(cmd)
        contour_nodes=[elem['id'] for elem in cursor.fetchall()]
        contours_df=get_nodes_data(cursor,contour_nodes)
        contours_df=contours_df.to_crs(osm_crs)
        if len(osm_edges_df)>0:
            local_intersection=gpd.overlay(contours_df,osm_edges_df,keep_geom_type=False).explode(index_parts=False)
            if len(local_intersection)>0:
                local_intersection['edge_coordinate']=local_intersection.apply(lambda row:osm_edges_df.loc[row['id_edge']]['geometry'].project(row['geometry']),axis=1)
            if intersection is None:
                intersection=local_intersection
                current_osm_edges=set(local_intersection['edge'])
            else:
                intersection=pd.concat([intersection,local_intersection],ignore_index=True)
                current_osm_edges=current_osm_edges.union(set(local_intersection['edge']))



            terminated_edges=current_osm_edges.difference(local_intersection[local_intersection.elevation>=max_elevation-elevation_cut]['edge'])
            if len(terminated_edges)>0:
                 osm_edges_df=osm_edges_df[~osm_edges_df['edge'].apply(lambda edge: edge in terminated_edges)]
            t3=time.time()
            print('%f<=elevation<%f :%f '%(min_elevation,max_elevation,t3-t2))
        else:
            break
    return intersection


#OSM NODES ELEVATION COMPUTATION


def estimate_elevations_from_laplacian(sub_G_osm):
    nodes=list(sub_G_osm.nodes())
    nx.set_edge_attributes(sub_G_osm,{(u,v,k):{'inverse_distance':1./d['length']} for u,v,k,d in sub_G_osm.edges(data=True,keys=True)})
    variable_indexes=[k for k,node in enumerate(nodes) if not(isinstance(node,tuple))]
    constant_indexes=[k for k,node in enumerate(nodes) if isinstance(node,tuple)]
    elevations=[sub_G_osm.nodes()[nodes[k]]['elevation'] for k in constant_indexes]
    if len(set(elevations))>1:
        L=nx.laplacian_matrix(sub_G_osm,weight='inverse_distance').toarray()
        A=np.array([[L[i,j] for j in variable_indexes] for i in variable_indexes])
        B=np.array([2*np.sum([L[i,constant_indexes[k]]*elevation for k,elevation in enumerate(elevations)]) for i in variable_indexes])

        K=np.linalg.norm(B)
        A/=K
        B/=K

        fun=lambda X:np.sum(X*np.matmul(A,X))+np.sum(B*X)
        x0=np.mean(elevations)*np.ones(len(variable_indexes))
        res=minimize(fun,x0)
        if res.success:
            return [nodes[k] for k in variable_indexes],res.x
        else:
            print(res.message)
    else:
        print('only one available elevation')



