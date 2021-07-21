#include "helpers.h"

int parseArguments (int argc, char *argv[], std::string & filename, unsigned & krylov_dim) {
        int c;

        while ((c = getopt (argc, argv, "k:f:")) != -1) {
                switch(c) {
                        case 'f':
                                filename=optarg; break;	 //Skip the CPU test
                        case 'k':
                                krylov_dim=atoi(optarg); break;	 //Skip the CPU test
                        default:
                                fprintf(stderr, "Invalid option given\n");
                                return -1;
                }
        }
        return 0;
}
