#include <iostream>
#include <fstream>
#include <cstddef>
#include <vector>

std::ofstream of;


std::string blue("#0072BD");
std::string red("#D85319");
std::string yellow("#FCD020");
std::string purple("#7E2F8E");
std::string green("#77AC30");

void dump_node(std::string const & name, int x,
               std::string const & color = "gray",
               std::string const & shape = "circle")
{
    of << name << x
       << " [label=\"\", shape=\"" << shape << "\""
       << " ratio=2 style=\"filled\" width=1.3 fillcolor=\""
       << color << "\"];"
       << std::endl;
}

void dump_node(std::string const & name, int x, int y,
               std::string const & color = "gray",
               std::string const & shape = "circle")
{
    of << name << x << y
       << " [label=\"\", shape=\"" << shape << "\""
       << " ratio=2 style=\"filled\" width=1.3 fillcolor=\""
       << color << "\"];"
       << std::endl;
}

void dump_node(std::string const & name, int x, int y, int z,
               std::string const & color = "gray",
               std::string const & shape = "circle")
{
    of << name << x << y << z
       << " [label=\"\", shape=\"" << shape << "\""
       << " ratio=2 style=\"filled\" width=1.3 fillcolor=\""
       << color << "\"];"
       << std::endl;
}



void dump_edge(std::string const & name1, int x,
               std::string const & name2,
               std::string const & color = "black")
{
    of << name1 << x << " -> "
       << name2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}


void dump_edge(std::string const & name1,
               std::string const & name2, int x,
               std::string const & color = "black")
{
    of << name1 << " -> "
       << name2 << x
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}


void dump_edge(std::string const & name1, int x1, int y1,
               std::string const & name2, int x2, int y2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << " -> "
       << name2 << x2 << y2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}


void dump_edge(std::string const & name1, int x1,
               std::string const & name2, int x2, int y2,
               std::string const & color = "black")
{
    of << name1 << x1 << " -> "
       << name2 << x2 << y2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}

void dump_edge(std::string const & name1, int x1, int y1,
               std::string const & name2, int x2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << " -> "
       << name2 << x2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}


void dump_edge(std::string const & name1, int x1, int y1,
               std::string const & name2, int x2, int y2, int z2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << " -> "
       << name2 << x2 << y2 << z2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}

void dump_edge(std::string const & name1, int x1, int y1, int z1,
               std::string const & name2, int x2, int y2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << z1 << " -> "
       << name2 << x2 << y2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}


void dump_edge(std::string const & name1, int x1, int y1, int z1,
               std::string const & name2, int x2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << z1 << " -> "
       << name2 << x2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}

void dump_edge(std::string const & name1, int x1, int y1, int z1,
               std::string const & name2, int x2, int y2, int z2,
               std::string const & color = "black")
{
    of << name1 << x1 << y1 << z1 << " -> "
       << name2 << x2 << y2 << z2
       << " [arrowsize=1 penwidth=2, color=\""
       << color << "\"];\n";
}



void dump_network_as_all_edges(std::vector<int> const& net)
{
    of << "digraph {\n";

    // dump forward nodes
    for ( int x = 0; x < net.size(); ++x )
    {
        if ( net[x] == 0 ) // POOLING NODES
        {
            int len = net[x-1];
            for ( int y = 0; y < len; ++y )
                dump_node("nout", x, y);
        }
        else
        {
            int len = net[x];
            for ( int y = 0; y < len; ++y )
            {
                if ( x > 0 )
                {
                    dump_node("nin", x, y);
                }

                dump_node("nout", x, y);

                if ( x > 0 )
                {
                    dump_edge("nin",x,y,"nout",x,y,green);
                }
            }
        }
    }

    // edges
    for ( int x = 1; x < net.size(); ++x )
    {
        if ( net[x] ) // FULLY CONNECTED
        {
            int len = net[x-1] ? net[x-1] : net[x-2];
            for ( int i = 0; i < len; ++i )
                for ( int j = 0; j < net[x]; ++j )
                    dump_edge("nout",x-1,i,"nin",x,j,red);
        }
        else // POOLING
        {
            int len = net[x-1];
            for ( int i = 0; i < len; ++i )
                dump_edge("nout",x-1,i,"nout",x,i,blue);
        }
    }

    of << "}\n";

}


void dump_network(std::vector<int> const& net)
{
    of << "digraph {\n";

    // dump forward nodes
    for ( int x = 0; x < net.size(); ++x )
    {
        int len = net[x] ? net[x] : net[x-1];
        std::string color = net[x] ? green : "gray";
        for ( int y = 0; y < len; ++y )
            dump_node("n", x, y, x==0 ? yellow : color);
    }

    // edges
    for ( int x = 1; x < net.size(); ++x )
    {
        if ( net[x] ) // FULLY CONNECTED
        {
            int len = net[x-1] ? net[x-1] : net[x-2];
            for ( int i = 0; i < len; ++i )
                for ( int j = 0; j < net[x]; ++j )
                    dump_edge("n",x-1,i,"n",x,j,red);
        }
        else // POOLING
        {
            int len = net[x-1];
            for ( int i = 0; i < len; ++i )
                dump_edge("n",x-1,i,"n",x,i,blue);
        }
    }

    of << "}\n";

}

void dump_task_dependencies(int fin, int fout, int bs)
{
    of << "digraph {\n";

    dump_node("sp", 1, yellow); // hack
    dump_node("sp", 2, yellow); // hack
    //dump_node("sp", 3, yellow); // hack
    //dump_node("sp", 4, yellow); // hack

    // Forward FFT images
    for ( int x = 0; x < fin; ++x )
        for ( int y = 0; y < bs; ++y )
        {
            dump_node("fwd_in_image",x,y,red);
            dump_edge("sp",1,"fwd_in_image",x,y);
            dump_edge("fwd_in_image",x,y,"sp",2);
        }

    // Backward FFT images
    for ( int x = 0; x < fout; ++x )
        for ( int y = 0; y < bs; ++y )
        {
            dump_node("bwd_out_image",x,y,purple);
            //dump_edge("sp3","bwd_out_image",x);
            //dump_edge("bwd_out_image",x,y,"sp",3);
        }

    for ( int x = 0; x < fout; ++x )
    {
        for ( int y = 0; y < fin; ++y )
        {
            dump_node("fwd_kernel",x,y,blue);

            //if ( y == 0 ) dump_edge("sp",2,"fwd_kernel",x,y);

            for ( int z = 0; z < bs; ++z )
            {
                dump_node("mad",x,y,z,green);
                dump_edge("fwd_kernel",x,y,"mad",x,y,z);

                if ( y == 0 )
                {
                    for ( int t = 0; t < fin; ++t )
                        dump_edge("fwd_in_image",t,z,"mad",x,y,z);
                }

                if ( y == (fin-1) ) dump_edge("mad",x,y,z,"bwd_out_image",x,z);
                else dump_edge("mad",x,y,z,"fwd_kernel",x,y+1);
            }
        }
    }

    of << "}\n";
}


int main()
{
    ///std::vector<int> v{1,5,0,5,5,2};

    // of.open("net.DOT");
    // dump_network(v);
    // of.close();

    // of.open("net2.DOT");
    // dump_network_as_all_edges(v);
    // of.close();

    of.open("deps.DOT");
    dump_task_dependencies(5,5,8);
    of.close();
}
