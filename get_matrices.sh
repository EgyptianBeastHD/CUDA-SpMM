#!/usr/bin/bash


cd matrices
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt120.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n24.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz

tar -xzf delaunay_n24.tar.gz && rm delaunay_n24.tar.gz
tar -xzf nlpkkt120.tar.gz && rm nlpkkt120.tar.gz
tar -xzf Cube_Coup_dt0.tar.gz && rm Cube_Coup_dt0.tar.gz
