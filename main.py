import osmnx as ox
import time
from multiprocessing import Pool,cpu_count
import glob
from lib_contour_lines import *
import mysql.connector


user='spirz'
password='this_is_my_PASSWORD_m8'
host='localhost'
database='dem_from_contours_lines'


contours_lines_table_name='contours_lines'
osm_nodes_table_name="osm_nodes"
osm_edges_table_name="osm_edges"
intersections_table_name="intersections"


place_name='Corse, France'
network_type='drive'

file_paths=glob.glob('COURBE_1-0__SHP_LAMB93_D02B_2021-01-01/COURBE/1_DONNEES_LIVRAISON_2021-01-01/COURBE_1-0_SHP_LAMB93_D02B_2021/*.shp')
file_paths+=glob.glob('COURBE_1-0__SHP_LAMB93_D02A_2021-01-01/COURBE/1_DONNEES_LIVRAISON_2021-01-01/COURBE_1-0_SHP_LAMB93_D02A_2021/*.shp')
elevation_column="ALTITUDE"

merging_distance=1.

n_bunch_edges=25000
elevation_step=10
elevation_cut=10.


if __name__ == "__main__":

	DBM=DataBaseManager(user=user,password=password,host=host,database=database,
	            contours_lines_table_name=contours_lines_table_name,
	            osm_nodes_table_name=osm_nodes_table_name,
	            osm_edges_table_name=osm_edges_table_name,
	            intersections_table_name=intersections_table_name)

	t1=time.time()
	DBM.delete_tables()
	DBM.create_tables_v1()
	DBM.create_tables_v2()
	t2=time.time()
	print('creating tables took %f'%(t2-t1))

	DBM.add_contour_lines_to_database(file_paths,elevation_column=elevation_column)
	t3=time.time()
	print('adding contour lines to database took %f'%(t3-t2))
	DBM.add_merged_contours_lines()
	t4=time.time()
	print('merging contour lines took %f'%(t4-t3))

	G_osm=ox.graph_from_place(place_name,network_type=network_type)
	G_osm,osm_crs=preprocess_osm_graph(G_osm)
	t5=time.time()
	print('loading and processing osm graph took %f'%(t5-t4))

	DBM.compute_all_intersections(G_osm,osm_crs,n_bunch_edges=n_bunch_edges,
							      elevation_step=elevation_step,elevation_cut=elevation_cut)
	t6=time.time()
	print('computing intersections took %f'%(t6-t5))

	G_osm_cut=DBM.build_cut_osm_graph()
	DBM.complete_osm_graph(G_osm_cut)
	DBM.add_nodes_elevations_to_database(G_osm_cut)
	t7=time.time()
	print('computing nodes elvations took %f'%(t7-t6))
	print('total %f'%(t7-t1))


