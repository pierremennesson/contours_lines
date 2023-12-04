import numpy as np
import networkx as nx
import geopandas as gpd
from copy import deepcopy
from itertools import combinations
from shapely.geometry import LineString

        

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
    closed_contour_lines,merged_nodes=[],set()
    for cycle in cycles:
        contour_line_coords=[]
        for k in range(len(cycle)-1):
            node_1,node_2=cycle[k],cycle[k+1]
            ls=G_extremities.get_edge_data(node_1,node_2)['geometry']
            coords=list(ls.coords)
            if node_1[0]==node_2[0] and node_1[1]==1:
                coords=coords[::-1]
            contour_line_coords+=coords
        closed_contour_lines.append(LineString(contour_line_coords))
        merged_nodes=merged_nodes.union(cycle)
    return closed_contour_lines,merged_nodes

    


