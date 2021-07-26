/**
 * \file:        read_undirected_graph_from_file.cc
 * \brief:       A function to populate the entries of adjMatrix.indexes. The matrix read in will be in upper triangular format, so the task is to change this to a full, symmetric matrix --- this is done to make multiplication faster at a later stage.
 * \author:      Hugh Delaney
 * \version:     1
 * \date:        2021-07-02
 */


/* --------------------------------------------------------------------------*/
/**
 * \brief:       
 *
 * \param:       std::string
 * \param:       A      A will have been initialized with n = N and indexes will have
 *                      dimension (2, 2*n);
 */
/* ----------------------------------------------------------------------------*/
void read_undirected_graph_from_file(std::string, adjMatrix & A) {

