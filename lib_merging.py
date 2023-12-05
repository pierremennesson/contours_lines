import numpy as np
import networkx as nx
import geopandas as gpd
from copy import deepcopy
from itertools import combinations
from shapely.geometry import LineString,MultiLineString

        

def build_open_contour_graph(level_open_contours_df,max_distance=1.,coeff_restriction=0.1):
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

def remove_branch(G_extremities):
    bad_edges=[]
    for node,deg in G_extremities.degree():
        if deg>2:
            edges=[edge for edge in G_extremities.edges(node,data=True) if edge[2]['edge_type']=='join']
            edges=sorted(edges,key=lambda edge:edge[2]['distance'])
            bad_edges+=[edge[:2] for edge in edges[1:]]
    G_extremities.remove_edges_from(bad_edges)

def cycle_chain_decomposition(G_extremities):
    cycles,chains=[],[]
    for cc in nx.connected_components(G_extremities):
        if len(cc)>0:
            sub_G=nx.subgraph(G_extremities,cc)
            cycle_basis=nx.cycle_basis(sub_G)
            if len(cycle_basis)==0:
                chain_extremities=[node for node,deg in sub_G.degree() if deg==1]
                assert len(chain_extremities)==2
                begin,end=chain_extremities
                chains.append(nx.shortest_path(sub_G,begin,end))
            else:
                assert len(cycle_basis)==1
                cycle=cycle_basis[0]
                cycle.append(cycle[0])
                cycles.append(cycle)
    return cycles,chains




def get_contour_lines_from_elevation_df(level_open_contours_df,max_distance=1):
    G_extremities=build_open_contour_graph(level_open_contours_df,max_distance=max_distance)
    clean_graph(G_extremities)
    remove_branch(G_extremities)
    cycles,paths=cycle_chain_decomposition(G_extremities)
    closed_contour_lines,open_contour_lines,merged_nodes=[],[],set()
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
        merged_nodes=merged_nodes.union([node[0] for node in cycle])
    for path in paths:
        contour_line_coords=[]
        for k in range(len(path)-1):
            node_1,node_2=path[k],path[k+1]
            ls=G_extremities.get_edge_data(node_1,node_2)['geometry']
            coords=list(ls.coords)
            if node_1[0]==node_2[0] and node_1[1]==1:
                coords=coords[::-1]
            contour_line_coords+=coords
        open_contour_lines.append(LineString(contour_line_coords))
        merged_nodes=merged_nodes.union([node[0] for node in path])
    return closed_contour_lines,open_contour_lines,merged_nodes

    


