

Point(1) = {0, 0, 0, 0.4} ;
Point(2) = {1, 0, 0, 0.04} ;
Point(3) = {0, 1, 0, 0.04} ;
Point(4) = {-1, 0, 0, 0.04} ;
Point(5) = {0, -1, 0, 0.04} ;

Circle(1) = {2, 1, 3} ;
Circle(2) = {3, 1, 4} ;
Circle(3) = {4, 1, 5} ;
Circle(4) = {5, 1, 2} ;

Line Loop(5) = {1, 2, 3, 4};

Surface(1) = {5} ;
Point{1} In Surface{1} ;
