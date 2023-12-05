import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from itertools import combinations
from shapely.geometry import Point,LineString
from shapely.wkt import loads
import time
from scipy.optimize import minimize
from lib_merging import get_contour_lines_from_elevation_df
import osmnx as ox

        
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


def preprocess_osm_graph(G_osm):
    G_osm=to_multi_graph(G_osm)
    osm_crs=ox.graph_to_gdfs(G_osm,edges=False).estimate_utm_crs()
    G_osm=ox.project_graph(G_osm,to_crs=osm_crs)
    add_missing_geometries(G_osm)
    return G_osm,osm_crs


#BUILD TABLES

def create_tables_v1(cursor,contours_lines_table_name='contours_lines'):
    cmd="""CREATE TABLE `%s`(
          `id` INT NOT NULL AUTO_INCREMENT,
          `source` INT,
          `is_closed` BOOLEAN NOT NULL,
          `elevation` FLOAT NOT NULL,
          `length` FLOAT,
          `number_points` FLOAT,
          `geometry` LINESTRING NOT NULL,
          PRIMARY KEY (id)
          )
          ENGINE=InnoDB;
        """%contours_lines_table_name
    cursor.execute(cmd)


def create_tables_v2(cursor,contours_lines_table_name='contours_lines',osm_nodes_table_name="osm_nodes",osm_edges_table_name="osm_edges",intersections_table_name="intersections"):

    cmd="""CREATE TABLE `%s`(
    `osm_id` BIGINT NOT NULL,
    `geometry` POINT NOT NULL,
    `elevation` FLOAT,

    PRIMARY KEY (osm_id)
    );
        """%osm_nodes_table_name
    
    cursor.execute(cmd)

    cmd="""CREATE TABLE `%s`(
    `osm_begin` BIGINT NOT NULL,
    `osm_end` BIGINT NOT NULL,
    `osm_key` INT NOT NULL,
    `length` FLOAT NOT NULL,
    `intersects` BOOLEAN,


    FOREIGN KEY (osm_begin) REFERENCES %s(osm_id),
    FOREIGN KEY (osm_end) REFERENCES %s(osm_id),
    PRIMARY KEY (osm_begin,osm_end,osm_key)

    );
        """%(osm_edges_table_name,osm_nodes_table_name,osm_nodes_table_name)
    
    cursor.execute(cmd)

    
    cmd="""CREATE TABLE `%s`(
        `id` INT NOT NULL AUTO_INCREMENT,
        `osm_begin` BIGINT NOT NULL,
        `osm_end` BIGINT NOT NULL,
        `osm_key` INT NOT NULL,
        `edge_coordinate` FLOAT NOT NULL,
        `contour_id` INT NOT NULL,
        `geometry` POINT NOT NULL,
        `elevation` FLOAT NOT NULL,

        FOREIGN KEY (contour_id) REFERENCES %s(id),
        FOREIGN KEY (osm_begin,osm_end,osm_key) REFERENCES %s(osm_begin,osm_end,osm_key),
        UNIQUE (osm_begin,osm_end,osm_key,edge_coordinate),
        PRIMARY KEY (id)

          )
          ENGINE=InnoDB;
        """%(intersections_table_name,contours_lines_table_name,osm_edges_table_name)
    cursor.execute(cmd)
    

def delete_tables(cursor,table_names=["contours_lines","osm_nodes","osm_edges","intersections"]):
    for table_name in table_names:
        cmd="DROP TABLE IF EXISTS %s"%table_name
        cursor.execute(cmd)
        
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
    for k,file_path in enumerate(file_paths):
        if (k+1)%5==0:
            print('file  %i/%i'%(k+1,len(file_paths)))
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
        bad_ids=[elem['id_2'] for elem in L if loads(elem['geometry_1'])==loads(elem['geometry_2'])]
        if len(bad_ids)>0:
            bad_ids_string='(%s)'%', '.join([str(bad_id) for bad_id in bad_ids])
            cmd="DELETE FROM %s WHERE id IN %s"%(contours_lines_table_name,bad_ids_string)
            cursor.execute(cmd)


        
#MERGING

def add_merged_contours_lines(cursor,contours_lines_table_name="contours_lines",merging_distance=1):
    cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%contours_lines_table_name
    cmd=cursor.execute(cmd)
    elevations=np.array([elem['elevation'] for elem in cursor.fetchall()])
    for k,level in enumerate(elevations):
        if (k+1)%10==0:
            print('elevation %i/%i'%(k+1,len(elevations)))
        level_open_contours_df=get_level_contours_df(cursor,level,contours_lines_table_name,is_closed=False)
        if level_open_contours_df is not None:
            crs=level_open_contours_df.estimate_utm_crs()
            level_open_contours_df=level_open_contours_df.to_crs(crs)
            closed_merged_contours_lines,open_merged_contour_lines,merged_nodes=get_contour_lines_from_elevation_df(level_open_contours_df,max_distance=merging_distance)
            if len(closed_merged_contours_lines)>0:
                closed_merged_contours_df=gpd.GeoDataFrame([{'geometry':ls} for ls in closed_merged_contours_lines],geometry='geometry',crs=crs)
                closed_merged_contours_df['elevation']=level 
                add_contour_lines_to_database_from_df(closed_merged_contours_df,cursor,is_closed=True)
            if len(open_merged_contour_lines)>0:
                open_merged_contours_df=gpd.GeoDataFrame([{'geometry':ls} for ls in open_merged_contour_lines],geometry='geometry',crs=crs)
                open_merged_contours_df['elevation']=level 
                add_contour_lines_to_database_from_df(open_merged_contours_df,cursor,is_closed=False)
                cursor.execute(cmd)
            if len(merged_nodes)>0:
                merge_nodes_string='(%s)'%', '.join([str(merge_node) for merge_node in merged_nodes])
                cmd="DELETE FROM %s WHERE id IN %s"%(contours_lines_table_name,merge_nodes_string)
                cursor.execute(cmd)


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
        contours_df=contours_df.rename(columns={'id':'contour_id'})
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



def add_intersection_to_database(cursor,intersection,edges,G_osm,osm_nodes_table_name="osm_nodes",osm_edges_table_name="osm_edges",intersections_table_name="intersections"):

    osm_nodes_data=nx.edge_subgraph(G_osm,edges).nodes(data=True)
    
    string_list=',\n'.join(['(%i,ST_GeomFromText(\'%s\'))'%(node,Point(data['lon'],data['lat']).wkt) for node,data in osm_nodes_data])
    cmd="INSERT IGNORE INTO %s(osm_id,geometry) VALUES %s;"%(osm_nodes_table_name,string_list)
    cursor.execute(cmd)

    intersecting_edges=intersection['edge'].unique()
    string_list=',\n'.join(['(%i,%i,%i,%f,TRUE)'%(edge[0],edge[1],edge[2],G_osm.get_edge_data(*edge)['length']) for edge in intersecting_edges])
    cmd="INSERT INTO %s(osm_begin,osm_end,osm_key,length,intersects) VALUES %s;"%(osm_edges_table_name,string_list)
    cursor.execute(cmd)

    inside_edges=set(edges).difference(intersecting_edges)

    string_list=',\n'.join(['(%i,%i,%i,%f,False)'%(edge[0],edge[1],edge[2],G_osm.get_edge_data(*edge)['length']) for edge in inside_edges if edge[0]!=edge[1]])
    cmd="INSERT INTO %s(osm_begin,osm_end,osm_key,length,intersects) VALUES %s;"%(osm_edges_table_name,string_list)
    cursor.execute(cmd)



    intersection=intersection.to_crs('epsg:4326')
    string_list=',\n'.join(['(%i,%i,%i,%f,%i,%f,ST_GeomFromText(\'%s\'))'%(row['edge'][0],row['edge'][1],row['edge'][2],row['edge_coordinate'],row['contour_id'],row['elevation'],row['geometry'].wkt)  for _,row in intersection.iterrows()])
    cmd="INSERT INTO %s (`osm_begin`,`osm_end`,`osm_key`,`edge_coordinate`,`contour_id`,`elevation`,`geometry`) VALUES %s;"%(intersections_table_name,string_list)
    cursor.execute(cmd)
    

def compute_all_intersections(G_osm,osm_crs,cursor,contours_lines_table_name="contours_lines",osm_nodes_table_name="osm_nodes",osm_edges_table_name="osm_edges",intersections_table_name="intersections",n_bunch_edges=20000,elevation_step=10,elevation_cut=10.):


    osm_edges_df=ox.graph_to_gdfs(G_osm,nodes=False,edges=True)
    osm_edges_df['edge']=osm_edges_df.index
    osm_edges_df=osm_edges_df.loc[:,['edge','geometry']]
    osm_edges_df=osm_edges_df.set_index(np.arange(len(osm_edges_df)))
    osm_edges_df['id_edge']=osm_edges_df.index

    for k in range(0,len(osm_edges_df),n_bunch_edges):
        sub_osm_edges_df=osm_edges_df.iloc[k:min(k+n_bunch_edges,len(osm_edges_df))]
        intersection=increasing_elevations_intersections(sub_osm_edges_df,osm_crs,cursor,contours_lines_table_name=contours_lines_table_name,elevation_step=elevation_step,elevation_cut=elevation_cut)
        add_intersection_to_database(cursor,intersection,sub_osm_edges_df['edge'],G_osm,osm_nodes_table_name=osm_nodes_table_name,osm_edges_table_name=osm_edges_table_name,intersections_table_name=intersections_table_name)










#OSM NODES ELEVATION COMPUTATION


def rebuild_osm_graph(cursor,osm_nodes_table_name="osm_nodes",osm_edges_table_name="osm_edges"):
    G_osm=nx.MultiGraph()
    cmd="SELECT osm_begin,osm_end,osm_key,length FROM %s WHERE NOT intersects"%osm_edges_table_name
    cursor.execute(cmd)
    G_osm.add_edges_from([(elem['osm_begin'],elem['osm_end'],elem['osm_key'],{'length':elem['length']}) for elem in cursor.fetchall()])
    cmd="SELECT osm_id,ST_asText(geometry) AS geometry FROM %s WHERE osm_id IN %s"%(osm_nodes_table_name,str(tuple(G_osm.nodes())))
    cursor.execute(cmd)
    data={elem['osm_id']:{'geometry':loads(elem['geometry'])} for elem in cursor.fetchall()}
    nx.set_node_attributes(G_osm,{node:{'x':datum['geometry'].x,'y':datum['geometry'].y} for node,datum in data.items()})
    return G_osm

def complete_osm_graph(G_osm,cursor,osm_edges_table_name="osm_edges",intersections_table_name="intersections"):
    true_osm_nodes=tuple(G_osm.nodes())
    k=0

    cmd="SELECT osm_begin,osm_end,osm_key,edge_coordinate,elevation,ST_asText(geometry) AS geometry FROM %s WHERE osm_begin IN %s"%(intersections_table_name,str(true_osm_nodes))
    cursor.execute(cmd)
    intersection=pd.DataFrame(cursor.fetchall())
    intersection['geometry']=intersection['geometry'].apply(lambda pt:loads(pt))

    for (osm_begin,osm_end,osm_key),df in intersection.groupby(['osm_begin','osm_end','osm_key']):
        df=df.sort_values('edge_coordinate')
        row=df.iloc[0]
        neighbor_pt,neighbor_elevation,length=row['geometry'],row['elevation'],row['edge_coordinate']
        G_osm.add_node((osm_begin,k),x=neighbor_pt.x,y=neighbor_pt.y,elevation=neighbor_elevation)
        G_osm.add_edge(osm_begin,(osm_begin,k),length=length)
        k+=1

    cmd="SELECT osm_begin,osm_end,osm_key,length FROM %s WHERE osm_end IN %s"%(osm_edges_table_name,str(true_osm_nodes))
    cursor.execute(cmd)
    lengths={(elem['osm_begin'],elem['osm_end'],elem['osm_key']):elem['length'] for elem in cursor.fetchall()}

    cmd="SELECT osm_begin,osm_end,osm_key,edge_coordinate,elevation,ST_asText(geometry) AS geometry FROM %s WHERE (osm_begin,osm_end,osm_key) IN %s"%(intersections_table_name,str(tuple(lengths.keys())))
    cursor.execute(cmd)
    intersection=pd.DataFrame(cursor.fetchall())
    intersection['geometry']=intersection['geometry'].apply(lambda pt:loads(pt))

    for (osm_begin,osm_end,osm_key),df in intersection.groupby(['osm_begin','osm_end','osm_key']):
        df=df.sort_values('edge_coordinate')
        row=df.iloc[-1]
        neighbor_pt,neighbor_elevation,length=row['geometry'],row['elevation'],lengths[(osm_begin,osm_end,osm_key)]-row['edge_coordinate']
        G_osm.add_node((osm_end,k),x=neighbor_pt.x,y=neighbor_pt.y,elevation=neighbor_elevation)
        G_osm.add_edge((osm_end,k),osm_end,length=length)
        k+=1


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


def add_nodes_elevations_to_database(G_osm,cursor,osm_nodes_table_name="osm_nodes"):
    for cc in nx.connected_components(nx.Graph(G_osm)):
        sub_G_osm=nx.subgraph(G_osm,cc)
        output=estimate_elevations_from_laplacian(sub_G_osm)
        if output is not None:
            nodes,estimated_elevations=output
            
            cmd="""CREATE TEMPORARY TABLE temp (
            `osm_id` BIGINT NOT NULL,
            `elevation` FLOAT NOT NULL
        
            );"""
            cursor.execute(cmd)
        
            cmd="INSERT INTO temp(`osm_id`,`elevation`) VALUES %s"%',\n'.join(['(%i,%f)'%(node,elevation) for node,elevation in zip(nodes,estimated_elevations)])
            cursor.execute(cmd)
        
            cmd=""" UPDATE %s JOIN temp
            ON %s.osm_id=temp.osm_id
            SET %s.elevation=temp.elevation
        
            ;"""%(osm_nodes_table_name,osm_nodes_table_name,osm_nodes_table_name)
            cursor.execute(cmd)
            
            cmd="DROP TABLE temp"
            cursor.execute(cmd)




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


#VISUALIZE

def get_altimetric_profile(G_osm,path,cursor,osm_nodes_table_name="osm_nodes",intersections_table_name="intersections"):
    edge_coordinates,elevations,nodes_coordinates,nodes_elevations=[],[],[],[]
    total_length=0
    for k in range(len(path)-1):
        cmd="SELECT elevation FROM %s WHERE osm_id=%i"%(osm_nodes_table_name,path[k])
        cursor.execute(cmd)
        elevation=cursor.fetchone()['elevation']
        if elevation is not None:
            edge_coordinates.append(total_length)
            elevations.append(elevation)
            nodes_coordinates.append(total_length)
            nodes_elevations.append(elevation)
        length=G_osm.get_edge_data(path[k],path[k+1],0)['length']
        cmd="SELECT elevation,edge_coordinate FROM %s WHERE osm_begin=%i AND osm_end=%i AND osm_key=0"%(intersections_table_name,path[k],path[k+1])
        cursor.execute(cmd)
        output=cursor.fetchall()
        if len(output)>0:
            output=sorted(output,key=lambda elem:elem['edge_coordinate'])
            edge_coordinates+=[total_length+elem['edge_coordinate'] for elem in output]
            elevations+=[elem['elevation'] for elem in output]
        else:
            cmd="SELECT elevation,edge_coordinate FROM %s WHERE osm_begin=%i AND osm_end=%i AND osm_key=0"%(intersections_table_name,path[k+1],path[k])
            cursor.execute(cmd)
            output=cursor.fetchall()
            if len(output)>0:
                output=sorted(output,key=lambda elem:-elem['edge_coordinate'])
                edge_coordinates+=[total_length+length-elem['edge_coordinate'] for elem in output]
                elevations+=[elem['elevation'] for elem in output]
        total_length+=length
    cmd="SELECT elevation FROM %s WHERE osm_id=%i"%(osm_nodes_table_name,path[-1])
    cursor.execute(cmd)
    elevation=cursor.fetchone()['elevation']
    if elevation is not None:
        edge_coordinates.append(total_length)
        elevations.append(elevation)
        nodes_coordinates.append(total_length)
        nodes_elevations.append(elevation)
    return edge_coordinates,elevations,nodes_coordinates,nodes_elevations