#include "helpers.h"

int parseArguments (int argc, char *argv[], std::string & filename, long unsigned & krylov_dim, bool & verbose) {
        int c;

        while ((c = getopt (argc, argv, "k:f:v")) != -1) {
                switch(c) {
                        case 'f':
                                filename=optarg; break;	 
                        case 'k':
                                krylov_dim=atoi(optarg); break;	 
                        case 'v':
                                verbose=true; break;
                        default:
                                fprintf(stderr, "Invalid option given\n");
                                return -1;
                }
        }
        return 0;
}
