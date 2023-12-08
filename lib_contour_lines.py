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
import mysql.connector
from multiprocessing import Pool,cpu_count



        

class DataBaseManager:
    """Main class to compute, store and retrieve elevation data. 
       Assumes that a database has been created and that user has been granted access.

    Parameters
    ----------
    user : user to connect to mysql database

    password : password to connect to mysql database

    host : host to connect to mysql database

    database : database name

    contours_lines_table_name : name of the tables that will store contour lines

    osm_nodes_table_name : name of the tables that will store open street map nodes

    osm_edges_table_name : name of the tables that will store open street map edges

    intersections_table_name : name of the tables that will store intersections between open street edges and contour lines


    """

    def __init__(self, user='spirz',password='this_is_my_PASSWORD_m8',host='localhost',database='dem_from_contours_lines',
        contours_lines_table_name='contours_lines',osm_nodes_table_name="osm_nodes",osm_edges_table_name="osm_edges",
        intersections_table_name="intersections"):


        self.user=user
        self.password=password
        self.host=host
        self.database=database
        self.cnx = mysql.connector.connect(user=self.user, 
                                  password=self.password,
                                  host=self.host,
                                  database=self.database,
                                  autocommit=True)

        self.cursor = self.cnx.cursor(buffered=True,dictionary=True)
        self.contours_lines_table_name = contours_lines_table_name
        self.osm_nodes_table_name = osm_nodes_table_name 
        self.osm_edges_table_name = osm_edges_table_name 
        self.intersections_table_name = intersections_table_name 
        self.config_connexion()

    def config_connexion(self):
        """ update mysql global variables
            to execute insert queries with
            a lot of rows

        """
        cmd="SET GLOBAL NET_BUFFER_LENGTH=1000000;"
        self.cursor.execute(cmd)
        cmd="SET GLOBAL MAX_ALLOWED_PACKET=1000000000;"
        self.cursor.execute(cmd)


    def reconnect(self,max_nb_try=100):
        """reconnect to database

        Parameters
        ----------
        max_nb_try : number of reconnexion trials


        """
        self.cursor.close()
        self.cnx.close()
        error,nb_try=True,0
        while(error and nb_try<max_nb_try):
            try:
                self.cnx = mysql.connector.connect(user=self.user, 
                                          password=self.password,
                                          host=self.host,
                                          database=self.database,
                                          autocommit=True)

                self.cursor = self.cnx.cursor(buffered=True,dictionary=True)
                self.config_connexion()
                print('reconnected !')
                error=False
            except Exception as e:
                nb_try+=1
                print(nb_try)
                time.sleep(5.)



    def execute(self,cmd):
        """execute mysql command

        Parameters
        ----------
        cmd : mysql command

        """
        try:
            self.cursor.execute(cmd)
        except Exception as e:
            self.reconnect()
            self.cursor.execute(cmd)

#BUILD TABLES

    def create_tables_v1(self):
        """ create mysql table that
            will store contours lines

        """
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
            """%self.contours_lines_table_name
        self.execute(cmd)


    def create_tables_v2(self):
        """ create mysql tables that
            will store open street map
            data

        """

        cmd="""CREATE TABLE `%s`(
        `osm_id` BIGINT NOT NULL,
        `geometry` POINT NOT NULL,
        `elevation` FLOAT,

        PRIMARY KEY (osm_id)
        );
            """%self.osm_nodes_table_name
        
        self.execute(cmd)

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
            """%(self.osm_edges_table_name,self.osm_nodes_table_name,self.osm_nodes_table_name)
        
        self.execute(cmd)

        
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
            """%(self.intersections_table_name,self.contours_lines_table_name,self.osm_edges_table_name)
        self.execute(cmd)
        

    def delete_tables(self,table_names=None):
        """ delete msyql tables

        """    
        if table_names is None:
            table_names=[self.intersections_table_name,self.osm_edges_table_name,self.osm_nodes_table_name,self.contours_lines_table_name]
        for table_name in table_names:
            cmd="DROP TABLE IF EXISTS %s"%table_name
            self.execute(cmd)
            
    #ADD TO DATABASE

    #assume geometry is in some utm crs
    def add_contour_lines_to_database_from_df(self,df,source=None,is_closed=None,nb_inserts=100,verbose=True):
        """add contours lines from a geodataframe to database

        Parameters
        ----------
        df : geodataframe containing contour lines

        source : either None or some int identifying the source geodataframe

        is_closed : if one already knows that all contour lines in the geodataframe are either 
        closed or open, the contours lines will be automatically stored with boolean attributes
        is_closed. If not provided, this feature is computed. 

        verbose : boolean, whether to or not to print intermediate times (for optimization purpose)

        """

        # self.update_buffer()
        t1=time.time()
        if is_closed is not None:
            df['is_closed']=is_closed
        else:
            df['is_closed']=df['geometry'].apply(lambda ls:ls.is_closed)
        df['length']=df['geometry'].apply(lambda ls:ls.length)
        df['number_points']=df['geometry'].apply(lambda ls:len(ls.coords))
        df=df.to_crs('epsg:4326')
        t2=time.time()
        if verbose:
            print('preprocess df took %f'%(t2-t1))
        if source is None:
            string_list=',\n'.join(['(%s,%f,%f,%f,ST_GeomFromText(\'%s\'))'%(str(row['is_closed']).upper(),row['elevation'],row['length'],row['number_points'],row['geometry'].wkt)  for _,row in df.iterrows()])
            cmd="""INSERT INTO %s (`is_closed`,`elevation`,`length`,`number_points`,`geometry`) VALUES %s;"""%(self.contours_lines_table_name,string_list)
        else:
            string_list=',\n'.join(['(%i,%s,%f,%f,%f,ST_GeomFromText(\'%s\'))'%(source,str(row['is_closed']).upper(),row['elevation'],row['length'],row['number_points'],row['geometry'].wkt)  for _,row in df.iterrows()])
            cmd="""INSERT INTO %s (`source`,`is_closed`,`elevation`,`length`,`number_points`,`geometry`) VALUES %s;"""%(self.contours_lines_table_name,string_list)
        self.execute(cmd)
        t3=time.time()
        if verbose:
            print('inserting %i rows took %f'%(len(df),t3-t2))
            

    def add_contour_lines_to_database(self,file_paths,elevation_column="ALTITUDE"):
        """add contours lines from a list of file_paths to geodataframes to database

        Parameters
        ----------
        file_paths : list of file_paths to geodataframes

        elevation_column : name of the geodataframes column corresponding to the elevation



        """
        for k,file_path in enumerate(file_paths):
            print('file  %i/%i'%(k+1,len(file_paths)))
            t1=time.time()
            df=gpd.read_file(file_path) 
            t2=time.time()
            df['elevation']=df[elevation_column]
            print('reading file took %s'%(t2-t1))
            self.add_contour_lines_to_database_from_df(df,source=k)



            
    #MERGING

    def add_merged_contours_lines(self,elevations=None,merging_distance=1):
        """merge contour lines that lie on a common elevation level, update the 
        database by adding the new contour lines and deleting the ones that have
        been merged.

        Parameters
        ----------
        elevations : list of elevation levels to perform the merging operation. If
        not provided, the merging is done for each elevation.

        merging_distance : see get_contour_lines_from_elevation_df in lib_merging


        """
        if elevations is None:
            # self.update_buffer()
            cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%self.contours_lines_table_name
            self.execute(cmd)
            elevations=np.array([elem['elevation'] for elem in self.cursor.fetchall()])
        for k,level in enumerate(elevations):
            if (k+1)%10==0:
                print('elevation %i/%i'%(k+1,len(elevations)))
            level_open_contours_df=self.get_level_contours_df(level,is_closed=False)
            if level_open_contours_df is not None:
                crs=level_open_contours_df.estimate_utm_crs()
                level_open_contours_df=level_open_contours_df.to_crs(crs)
                closed_merged_contours_lines,open_merged_contour_lines,merged_nodes=get_contour_lines_from_elevation_df(level_open_contours_df,max_distance=merging_distance)
                if len(closed_merged_contours_lines)>0:
                    closed_merged_contours_df=gpd.GeoDataFrame([{'geometry':ls} for ls in closed_merged_contours_lines],geometry='geometry',crs=crs)
                    closed_merged_contours_df['elevation']=level 
                    self.add_contour_lines_to_database_from_df(closed_merged_contours_df,is_closed=True,verbose=False)
                if len(open_merged_contour_lines)>0:
                    open_merged_contours_df=gpd.GeoDataFrame([{'geometry':ls} for ls in open_merged_contour_lines],geometry='geometry',crs=crs)
                    open_merged_contours_df['elevation']=level 
                    self.add_contour_lines_to_database_from_df(open_merged_contours_df,is_closed=False,verbose=False)
                if len(merged_nodes)>0:
                    merge_nodes_string='(%s)'%', '.join([str(merge_node) for merge_node in merged_nodes])
                    cmd="DELETE FROM %s WHERE id IN %s"%(self.contours_lines_table_name,merge_nodes_string)
                    self.execute(cmd)


    #INTERSECTION


    def increasing_elevations_intersections(self,osm_edges_df,osm_crs,elevation_step=10,elevation_cut=10.,drop_duplicates=False):
        """This functions recursively computes intersections between the contour_lines and the edges of some
        open street map graph. 


        Parameters
        ----------
        osm_edges_df : a geodataframe containing the edges of the open street map graph

        osm_crs : the coordinates reference system of osm_edges_df

        elevation_step : the intersections are computed by intersecting the osm_edges_df with the contour lines. 
        If the memory space is limited, this intersetion can not be computed directly. 
        To remedy this situation, we sort the distinct elevations, group them by bunch of size elevation_step and 
        progressively compute the intersections with the osm edges.

        elevation_cut : see update_local_intersection

        drop_duplicates : see update_local_intersection

        Returns
        -------
        intersection : a geodataframe whose rows are points that correspond to intersections between contour lines and osm edges.
        """
        t1=time.time()
        cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%self.contours_lines_table_name
        cmd=self.execute(cmd)
        elevations=np.array([elem['elevation'] for elem in self.cursor.fetchall()])

        intersection,current_osm_edges=None,None
        for i in range(0,len(elevations)-elevation_step,elevation_step):
            t2=time.time()
            min_elevation,max_elevation=elevations[i],elevations[i+elevation_step]
            cmd="SELECT id FROM %s WHERE elevation>=%f AND elevation<%i"%(self.contours_lines_table_name,min_elevation,max_elevation)
            self.execute(cmd)
            contour_nodes=[elem['id'] for elem in self.cursor.fetchall()]
            contours_df=self.get_nodes_data(contour_nodes)
            contours_df=contours_df.rename(columns={'id':'contour_id'})
            contours_df=contours_df.to_crs(osm_crs)
            if len(osm_edges_df)>0:
                intersection,osm_edges_df,current_osm_edges=update_local_intersection(intersection,current_osm_edges,contours_df,osm_edges_df,max_elevation,elevation_cut,drop_duplicates=drop_duplicates)
                t3=time.time()
                print('%f<=elevation<%f :%f '%(min_elevation,max_elevation,t3-t2))
            else:
                break
        return intersection



    def add_intersection_to_database(self,intersection,edges,G_osm):
        """This functions adds the intersection between contour lines 
        and osm edges to database.


        Parameters
        ----------
        intersection : a geodataframe whose rows are points that correspond 
        to intersections between contour lines and osm edges.

        edges : the osm edges. In order to compute nodes elevations, we need to store 
        all osm all edges, not only the ones intersecting contour lines that appear 
        in intersection.

        G_osm : the osm graph, needed to store osm edges length in the database.

        """
        # self.update_buffer()


        osm_nodes_data=nx.edge_subgraph(G_osm,edges).nodes(data=True)
        
        string_list=',\n'.join(['(%i,ST_GeomFromText(\'%s\'))'%(node,Point(data['lon'],data['lat']).wkt) for node,data in osm_nodes_data])
        cmd="INSERT IGNORE INTO %s(osm_id,geometry) VALUES %s;"%(self.osm_nodes_table_name,string_list)
        self.execute(cmd)

        intersecting_edges=intersection['edge'].unique()
        string_list=',\n'.join(['(%i,%i,%i,%f,TRUE)'%(edge[0],edge[1],edge[2],G_osm.get_edge_data(*edge)['length']) for edge in intersecting_edges])
        cmd="INSERT INTO %s(osm_begin,osm_end,osm_key,length,intersects) VALUES %s;"%(self.osm_edges_table_name,string_list)
        self.execute(cmd)

        inside_edges=set(edges).difference(intersecting_edges)

        string_list=',\n'.join(['(%i,%i,%i,%f,False)'%(edge[0],edge[1],edge[2],G_osm.get_edge_data(*edge)['length']) for edge in inside_edges if edge[0]!=edge[1]])
        cmd="INSERT INTO %s(osm_begin,osm_end,osm_key,length,intersects) VALUES %s;"%(self.osm_edges_table_name,string_list)
        self.execute(cmd)



        intersection=intersection.to_crs('epsg:4326')
        string_list=',\n'.join(['(%i,%i,%i,%f,%i,%f,ST_GeomFromText(\'%s\'))'%(row['edge'][0],row['edge'][1],row['edge'][2],row['edge_coordinate'],row['contour_id'],row['elevation'],row['geometry'].wkt)  for _,row in intersection.iterrows()])
        cmd="INSERT INTO %s (`osm_begin`,`osm_end`,`osm_key`,`edge_coordinate`,`contour_id`,`elevation`,`geometry`) VALUES %s;"%(self.intersections_table_name,string_list)
        self.execute(cmd)
        

    def compute_all_intersections(self,G_osm,osm_crs,n_bunch_edges=25000,elevation_step=10,elevation_cut=10.,drop_duplicates=False):
        """This functions recursively the contour_lines and the edges of some open street map graph and store
        them in the database. 


        Parameters
        ----------
        G_osm : an open street map graph

        osm_crs : the coordinates reference system of G_osm

        n_bunch_edges : the intersections are computed by intersecting the osm_edges_df with the contour lines. 
        If the memory space is limited, this intersetion can not be computed directly and osm edges are grouped in 
        bunch of size n_bunch_edges (contour lines are grouped by bunch determined by elevation_step see increasing_elevations_intersections).

        elevation_step : see increasing_elevations_intersections

        elevation_cut : see update_local_intersection

        drop_duplicates : see update_local_intersection

        """
        osm_edges_df=ox.graph_to_gdfs(G_osm,nodes=False,edges=True)
        osm_edges_df['edge']=osm_edges_df.index
        osm_edges_df=osm_edges_df.loc[:,['edge','geometry']]
        osm_edges_df=osm_edges_df.set_index(np.arange(len(osm_edges_df)))
        osm_edges_df['id_edge']=osm_edges_df.index

        for k in range(0,len(osm_edges_df),n_bunch_edges):
            sub_osm_edges_df=osm_edges_df.iloc[k:min(k+n_bunch_edges,len(osm_edges_df))]
            intersection=self.increasing_elevations_intersections(sub_osm_edges_df,osm_crs,elevation_step=elevation_step,elevation_cut=elevation_cut,drop_duplicates=drop_duplicates)
            self.add_intersection_to_database(intersection,sub_osm_edges_df['edge'],G_osm)










    #OSM NODES ELEVATION COMPUTATION


    def build_cut_osm_graph(self):
        """This functions builds the
        osm graph whose intersecting 
        edges removed
        
        Returns
        -------
        G_osm_cut : the cut osm graph
        """
        G_osm_cut=nx.MultiGraph()
        cmd="SELECT osm_begin,osm_end,osm_key,length FROM %s WHERE NOT intersects"%self.osm_edges_table_name
        self.execute(cmd)
        G_osm_cut.add_edges_from([(elem['osm_begin'],elem['osm_end'],elem['osm_key'],{'length':elem['length']}) for elem in self.cursor.fetchall()])
        cmd="SELECT osm_id,ST_asText(geometry) AS geometry FROM %s WHERE osm_id IN %s"%(self.osm_nodes_table_name,str(tuple(G_osm_cut.nodes())))
        self.execute(cmd)
        data={elem['osm_id']:{'geometry':loads(elem['geometry'])} for elem in self.cursor.fetchall()}
        nx.set_node_attributes(G_osm_cut,{node:{'x':datum['geometry'].x,'y':datum['geometry'].y} for node,datum in data.items()})
        return G_osm_cut

    def complete_osm_graph(self,G_osm_cut):
        """This functions add some nodes
        to the cut graph.
        For each node, we consider old intersecting
        edges with this node as extremity, add the
        closest contour line-edge intersection edgewise 
        to the graph and connect it to the node.


        Parameters
        ----------
        G_osm_cut : the osm cut graph

        """
        true_osm_nodes=tuple(G_osm_cut.nodes())
        k=0

        cmd="SELECT osm_begin,osm_end,osm_key,edge_coordinate,elevation,ST_asText(geometry) AS geometry FROM %s WHERE osm_begin IN %s"%(self.intersections_table_name,str(true_osm_nodes))
        self.execute(cmd)
        intersection=pd.DataFrame(self.cursor.fetchall())
        intersection['geometry']=intersection['geometry'].apply(lambda pt:loads(pt))

        for (osm_begin,osm_end,osm_key),df in intersection.groupby(['osm_begin','osm_end','osm_key']):
            df=df.sort_values('edge_coordinate')
            row=df.iloc[0]
            neighbor_pt,neighbor_elevation,length=row['geometry'],row['elevation'],row['edge_coordinate']
            G_osm_cut.add_node((osm_begin,k),x=neighbor_pt.x,y=neighbor_pt.y,elevation=neighbor_elevation)
            G_osm_cut.add_edge(osm_begin,(osm_begin,k),length=length)
            k+=1

        cmd="SELECT osm_begin,osm_end,osm_key,length FROM %s WHERE osm_end IN %s"%(self.osm_edges_table_name,str(true_osm_nodes))
        self.execute(cmd)
        lengths={(elem['osm_begin'],elem['osm_end'],elem['osm_key']):elem['length'] for elem in self.cursor.fetchall()}

        cmd="SELECT osm_begin,osm_end,osm_key,edge_coordinate,elevation,ST_asText(geometry) AS geometry FROM %s WHERE (osm_begin,osm_end,osm_key) IN %s"%(self.intersections_table_name,str(tuple(lengths.keys())))
        self.execute(cmd)
        intersection=pd.DataFrame(self.cursor.fetchall())
        intersection['geometry']=intersection['geometry'].apply(lambda pt:loads(pt))

        for (osm_begin,osm_end,osm_key),df in intersection.groupby(['osm_begin','osm_end','osm_key']):
            df=df.sort_values('edge_coordinate')
            row=df.iloc[-1]
            neighbor_pt,neighbor_elevation,length=row['geometry'],row['elevation'],lengths[(osm_begin,osm_end,osm_key)]-row['edge_coordinate']
            G_osm_cut.add_node((osm_end,k),x=neighbor_pt.x,y=neighbor_pt.y,elevation=neighbor_elevation)
            G_osm_cut.add_edge((osm_end,k),osm_end,length=length)
            k+=1



    def add_nodes_elevations_to_database(self,G_osm_cut):
        """This functions estimate each nodes elevations
        to the graph and connect it to the node.

        Parameters
        ----------
        G_osm_cut : the osm cut graph

        """
#        self.update_buffer()
        for cc in nx.connected_components(nx.Graph(G_osm_cut)):
            sub_G_osm_cut=nx.subgraph(G_osm_cut,cc)
            output=estimate_elevations_from_laplacian(sub_G_osm_cut)
            if output is not None:
                nodes,estimated_elevations=output
                
                cmd="""CREATE TEMPORARY TABLE temp (
                `osm_id` BIGINT NOT NULL,
                `elevation` FLOAT NOT NULL
            
                );"""
                self.execute(cmd)
            
                cmd="INSERT INTO temp(`osm_id`,`elevation`) VALUES %s"%',\n'.join(['(%i,%f)'%(node,elevation) for node,elevation in zip(nodes,estimated_elevations)])
                self.execute(cmd)
            
                cmd=""" UPDATE %s JOIN temp
                ON %s.osm_id=temp.osm_id
                SET %s.elevation=temp.elevation
            
                ;"""%(self.osm_nodes_table_name,self.osm_nodes_table_name,self.osm_nodes_table_name)
                self.execute(cmd)
                
                cmd="DROP TABLE temp"
                self.execute(cmd)


    #CONTOUR LINES GRAPH INFERENCE


    def build_contour_graph(self):
        """This functions builds the
        contours lines graph induced by
        successive intersections
        
        Returns
        -------
        G_contours : the contours lines graph 
        """
        G_contours=nx.Graph()
        cmd="SELECT osm_begin,osm_end,osm_key,edge_coordinate,contour_id,ST_asText(geometry) AS geometry FROM %s"%self.intersections_table_name
        self.execute(cmd)
        df=pd.DataFrame(self.cursor.fetchall())
        df['geometry']=df['geometry'].apply(lambda pt:loads(pt))
        df['edge']=df.apply(lambda row:(row['osm_begin'],row['osm_end'],row['osm_key']),axis=1)
        for _,sub_df in df.groupby('edge'):
            sub_df=sub_df.sort_values('edge_coordinate')
            for k in range(len(sub_df)-1):
                G_contours.add_edge(sub_df.iloc[k]['contour_id'],sub_df.iloc[k+1]['contour_id'])


        return G_contours,df

    #ACCESS DATA

    def get_level_contours_df(self,level,is_closed=True):
        """This functions retrieves the contour lines at
        some elevation level and structure them in a geodataframe.

        Parameters
        ----------
        level : the elevation level

        is_closed : whether to or not to retrieve only closed 
        contour lines, if set to None, all contour lines are 
        retrieved.

        """
        cmd="SELECT id,is_closed,elevation,ST_asText(geometry) AS geometry FROM %s WHERE elevation=%f" %(self.contours_lines_table_name,level)
        if is_closed is not None:
            if is_closed:
                cmd=cmd+" AND is_closed"
            else:
                cmd=cmd+" AND NOT is_closed"
        self.execute(cmd)
        level_open_contours_df=pd.DataFrame(self.cursor.fetchall())
        if len(level_open_contours_df)==0:
            return None
        level_open_contours_df['geometry']=level_open_contours_df['geometry'].apply(lambda ls:loads(ls))
        level_open_contours_df=level_open_contours_df.set_index('id',drop=False)
        level_open_contours_df=gpd.GeoDataFrame(level_open_contours_df,geometry='geometry',crs='epsg:4326')
        return level_open_contours_df

    def get_nodes_data(self,contour_line_ids):
        """This functions retrieves 
        the contour lines from their
        ids in the databases
        some elevation level and structure them in a geodataframe.

        Parameters
        ----------
        contour_line_ids : the contour lines ids in 
        the databases.

        """
        nodes_list='(%s)'%','.join([str(contour_line_id) for contour_line_id in contour_line_ids])
        cmd="SELECT id,elevation,number_points,ST_asText(geometry) AS geometry FROM %s WHERE id IN %s" %(self.contours_lines_table_name,nodes_list)
        self.execute(cmd)
        data= self.cursor.fetchall()
        data=pd.DataFrame(data)
        data['geometry']=loads(data['geometry'])
        data=gpd.GeoDataFrame(data,geometry='geometry',crs='epsg:4326')

        return data


    #VISUALIZE

    def get_altimetric_profile(self,G_osm,path):
        """This retrieves the altimetric profile 
        along a path in the osm graph.


        Parameters
        ----------
        G_osm : the osm graph whose edges intersections
        have been computed

        path : a path in G_osm

        Returns
        -------
        edge_coordinates : the x coordinates of the
        points in the altimetric profile

        elevations : the y coordinates of the
        points in the altimetric profile

        nodes_coordinates : the x coordinates of the
        nodes in the path


        nodes_elevations : the y coordinates of the
        nodes in the path
        """
        edge_coordinates,elevations,nodes_coordinates,nodes_elevations=[],[],[],[]
        total_length=0
        for k in range(len(path)-1):
            cmd="SELECT elevation FROM %s WHERE osm_id=%i"%(self.osm_nodes_table_name,path[k])
            self.execute(cmd)
            elevation=self.cursor.fetchone()['elevation']
            if elevation is not None:
                edge_coordinates.append(total_length)
                elevations.append(elevation)
                nodes_coordinates.append(total_length)
                nodes_elevations.append(elevation)
            length=G_osm.get_edge_data(path[k],path[k+1],0)['length']
            cmd="SELECT elevation,edge_coordinate FROM %s WHERE osm_begin=%i AND osm_end=%i AND osm_key=0"%(self.intersections_table_name,path[k],path[k+1])
            self.execute(cmd)
            output=self.cursor.fetchall()
            if len(output)>0:
                output=sorted(output,key=lambda elem:elem['edge_coordinate'])
                edge_coordinates+=[total_length+elem['edge_coordinate'] for elem in output]
                elevations+=[elem['elevation'] for elem in output]
            else:
                cmd="SELECT elevation,edge_coordinate FROM %s WHERE osm_begin=%i AND osm_end=%i AND osm_key=0"%(self.intersections_table_name,path[k+1],path[k])
                self.execute(cmd)
                output=self.cursor.fetchall()
                if len(output)>0:
                    output=sorted(output,key=lambda elem:-elem['edge_coordinate'])
                    edge_coordinates+=[total_length+length-elem['edge_coordinate'] for elem in output]
                    elevations+=[elem['elevation'] for elem in output]
            total_length+=length
        cmd="SELECT elevation FROM %s WHERE osm_id=%i"%(self.osm_nodes_table_name,path[-1])
        self.execute(cmd)
        elevation=self.cursor.fetchone()['elevation']
        if elevation is not None:
            edge_coordinates.append(total_length)
            elevations.append(elevation)
            nodes_coordinates.append(total_length)
            nodes_elevations.append(elevation)
        return edge_coordinates,elevations,nodes_coordinates,nodes_elevations


#UTILS INTERSECTION

def update_local_intersection(intersection,current_osm_edges,contours_df,osm_edges_df,max_elevation,elevation_cut,drop_duplicates=False):
    """This functions adds new intersections and clear edges that have been fully processed.


    Parameters
    ----------
    osm_edges_df : a geodataframe containing the current intersections between contour lines and osm edges

    current_osm_edges : the edges that have intersected so far

    contours_df : a geodataframe containing the new contour lines

    osm_edges_df : a geodataframe containing the osm edges

    max_elevation : the last elevation level appearing in contours_df


    elevation_cut : If the contour lines are "good", given an osm edge that intersect some contour line at elevation<max_elevation
    but not contour lines at max_elevation, then it should not intersect any elevation>max_elevation.
    This hypothesis lets us progressively discard some osm edges in the intersection computations to fasten computations and elevation_cut 
    controls the confidence we have : contour lines intersecting some elevation elevation<=max_elevation but no elevation in 
    ]max_elevation-elevation_cut,max_elevation] will be discarded.

    drop_duplicates : see if the contours lines have not been merged properly, one edge will intersect the same physical contour line
    several times and one should drop duplicate intersections. Should be set to True if lines have not been merged but to False if so
    to handle bugs caused by the UNIQUE (osm_begin,osm_end,osm_key,edge_coordinate) condition on the inersections table.


    Returns
    -------
    intersection : the updated intersection

    osm_edges_df : the (possibly filtered) osm edges geodataframe

    current_osm_edges : the updated list of intersecting eges
    """
    if len(osm_edges_df)>0:
        local_intersection=gpd.overlay(contours_df,osm_edges_df,keep_geom_type=False).explode(index_parts=False)

        if len(local_intersection)>0:
            local_intersection['edge_coordinate']=local_intersection.apply(lambda row:osm_edges_df.loc[row['id_edge']]['geometry'].project(row['geometry']),axis=1)
            if drop_duplicates:
                local_intersection=local_intersection.drop_duplicates(subset=['edge','edge_coordinate'])
        if intersection is None:
            intersection=local_intersection
            current_osm_edges=set(local_intersection['edge'])
        else:
            intersection=pd.concat([intersection,local_intersection],ignore_index=True)
            current_osm_edges=current_osm_edges.union(set(local_intersection['edge']))

        terminated_edges=current_osm_edges.difference(local_intersection[local_intersection.elevation>=max_elevation-elevation_cut]['edge'])
        if len(terminated_edges)>0:
             osm_edges_df=osm_edges_df[~osm_edges_df['edge'].apply(lambda edge: edge in terminated_edges)]

        return intersection,osm_edges_df,current_osm_edges

#UTILS NODES ELEVATIONS

def estimate_elevations_from_laplacian(sub_G_osm):
    """This functions estimate the nodes elevations
    using a constrained version of the laplacian.


    Parameters
    ----------
    sub_G_osm : a connected component of the cut graph


    Returns
    -------
    The nodes along with their estimated elevations or
    None if the optimization wasn't successful.

    """
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
    """This function preprocess the
    osm MultiDiGraph : carefully looses
    the orientation, add missing geometries
    and compute the coordinates reference
    system.

    Parameters
    ----------
    G_osm : an osm multidigraph


    Returns
    -------
    G_osm : the preprocessed multigraph

    crs : the coordinates reference system

    """
    G_osm=to_multi_graph(G_osm)
    osm_crs=ox.graph_to_gdfs(G_osm,edges=False).estimate_utm_crs()
    G_osm=ox.project_graph(G_osm,to_crs=osm_crs)
    add_missing_geometries(G_osm)
    return G_osm,osm_crs


#MUTLIPROCESS
def chunk(L,nb_cpu):
    """This function splits a list 
    in list of lists to be multiprocessed

    Parameters
    ----------
    L : an iterable

    nb_cpu : the number of list of lists, should
    be equal to the number of avalaible cpus


    Returns
    -------
    list of lists of files paths



    """
    N=len(L)
    n=max(N//nb_cpu,1)
    Ls=[L[i:i+n] for i in range(0,N,n)]
    if len(Ls)>nb_cpu:
        last=Ls[-1]
        Ls=Ls[:-1]
        for k,elem in enumerate(last):
            Ls[k].append(elem)
    return Ls


def _add_merged_contours_lines(elevations,
                                   merging_distance=1,
                                   user='spirz',
                                   password='this_is_my_PASSWORD_m8',
                                   host='localhost',
                                   database='dem_from_contours_lines',
                                   contours_lines_table_name='contours_lines'):
    DBM=DataBaseManager(user=user,password=password,host=host,database=database,
                        contours_lines_table_name=contours_lines_table_name)
    DBM.add_merged_contours_lines(elevations=elevations,merging_distance=merging_distance)
    DBM.cursor.close()
    DBM.cnx.close()
    
def add_merged_contours_lines_multiprocess(nb_cpu=None,
                                   merging_distance=1,
                                   user='spirz',
                                   password='this_is_my_PASSWORD_m8',
                                   host='localhost',
                                   database='dem_from_contours_lines',
                                   contours_lines_table_name='contours_lines'):

    cnx = mysql.connector.connect(user=user, 
                              password=password,
                              host=host,
                              database=database,
                              autocommit=True)

    if nb_cpu is None:
        nb_cpu=cpu_count()-1
        print('%i cpus working'%nb_cpu)
    cursor = cnx.cursor(buffered=True,dictionary=True)
    cmd="SELECT DISTINCT elevation FROM %s ORDER BY elevation"%contours_lines_table_name
    cursor.execute(cmd)
    elevations=[elem['elevation'] for elem in cursor.fetchall()]
    cursor.close()
    cnx.close()

    list_of_elevations=chunk(elevations,nb_cpu)
    args=[(bunch_of_elevations,merging_distance,user,password,host,database,contours_lines_table_name) for bunch_of_elevations in list_of_elevations]
    with Pool(nb_cpu) as p:
        p.starmap(_add_merged_contours_lines,args)


