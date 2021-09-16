
/**
 * \file:        helpers.cu
 * \brief:       A few helper functions
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */

#include "helpers.h"

int parseArguments (int argc, char *argv[], std::string & filename, long unsigned & krylov_dim, bool & verbose, long unsigned & n, long unsigned & bar_deg, long unsigned & E) {
        int c;

        while ((c = getopt (argc, argv, "k:f:b:n:e:v")) != -1) {
                switch(c) {
                        case 'f':
                                filename=optarg; break;	 
                        case 'k':
                                krylov_dim=atoi(optarg); break;	 
                        case 'b':
                                bar_deg=atoi(optarg); break;	 
                        case 'n':
                                n=atoi(optarg); break;	 
                        case 'e':
                                E=atoi(optarg); break;	 
                        case 'v':
                                verbose=true; break;
                        default:
                                fprintf(stderr, "Invalid option given\n");
                                return -1;
                }
        }
        return 0;
}
