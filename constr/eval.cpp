#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_cubic_extension.hpp"
#include "../src/poseidon_goldilocks.hpp"

using namespace std;
using json = nlohmann::json;

void file2json(const string &fileName, json &j);
Goldilocks::Element fill_trace(uint64_t i, uint64_t j, uint64_t k);
Goldilocks::Element fill_variable(uint64_t i, uint64_t j);

//
// Evaluate the AST
// Note: This is a reference sequential implementation not optimized for performance
//
int main(int argc, char *argv[])
{
    //
    // Input arguments
    //
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <AST.json> <logN>" << endl;
        return 1;
    }
    string fileName = argv[1];
    uint64_t N = 1 << stoi(argv[2]);

    //
    //  Load AST
    //
    json ast;
    file2json(fileName, ast);
    auto metadata = ast["metadata"];
    auto expressions = ast["expressions"];
    auto nodes = ast["nodes"];

    //
    // Allocate memory for TRACE, VARIABLES, NODES_RES and EXP_EVALS
    //
    int degree = metadata["field"]["extension"]["degree"];
    uint64_t nSegments = metadata["trace_widths"].size();
    std::vector<uint64_t> traceWidths = metadata["trace_widths"].get<std::vector<uint64_t>>();
    vector<Goldilocks::Element *> trace(nSegments);
    for (uint64_t i = 0; i < nSegments; i++)
    {
        if (traceWidths[i] != 0)
        {
            trace[i] = new Goldilocks::Element[traceWidths[i] * N];
        }
        else
        {
            trace[i] = nullptr;
        }
    }

    uint64_t nVarGroups = metadata["num_variables"].size();
    std::vector<uint64_t> varGroupSizes = metadata["num_variables"].get<std::vector<uint64_t>>();
    vector<Goldilocks::Element *> variables(nVarGroups);

    for (uint64_t i = 0; i < nVarGroups; i++)
    {
        if (varGroupSizes[i] != 0)
        {
            variables[i] = new Goldilocks::Element[varGroupSizes[i]];
        }
        else
        {
            variables[i] = nullptr;
        }
    }
    uint64_t nExpressions = expressions.size();
    uint64_t expDim = 0;
    for (uint64_t i = 0; i < nExpressions; i++)
    {
        auto &expression = expressions[i];
        uint64_t node_id = expression["node_id"];
        uint64_t dim = nodes[node_id]["value"] == "base" ? 1 : degree;
        expDim += dim;
    }
    Goldilocks::Element *expEvals = new Goldilocks::Element[expDim * N];

    uint64_t nNodes = nodes.size();
    Goldilocks::Element *nodesRes = new Goldilocks::Element[nNodes * degree];

    //
    // Initialize TRACE and VARIABLES with deterministic inputs and initialize EXP_EVALS to 0
    //
    for (uint64_t i = 0; i < nSegments; i++)
    {
        for (uint64_t j = 0; j < N; ++j)
        {
            for (uint64_t k = 0; k < traceWidths[i]; ++k)
            {
                trace[i][j * traceWidths[i] + k] = fill_trace(i, j, k);
            }
        }
    }
    for (uint64_t i = 0; i < nVarGroups; i++)
    {
        for (uint64_t j = 0; j < varGroupSizes[i]; ++j)
        {
            variables[i][j] = fill_variable(i, j);
        }
    }
    for (uint64_t i = 0; i < N * expDim; i++)
    {
        expEvals[i] = Goldilocks::zero();
    }

    //
    // Evaluate the expressions
    //
    uint64_t nsums = 0;
    uint64_t nsubs = 0;
    uint64_t nmuls = 0;
    uint64_t nreads = 0;
    uint64_t nwrites = 0;

    // note: this loop is parallelizable! (if implemented properly)
    for (uint64_t irow = 0; irow < N; ++irow)
    {
        for (uint64_t i = 0; i < nNodes * 3; i++)
        {
            nodesRes[i] = Goldilocks::zero();
        }

        for (uint64_t k = 0; k < nNodes; k++)
        {
            auto &node = nodes[k];
            if (node["type"] == "var")
            {
                // node data
                uint64_t group = node["args"]["group"];
                uint64_t offset = node["args"]["offset"];
                uint64_t dim = node["value"] == "base" ? 1 : degree;

                // asserts
                assert(group >= 0 && group < nVarGroups);
                assert(offset >= 0 && offset < varGroupSizes[group]);

                // copy
                memcpy(&nodesRes[k * degree], &variables[group][offset], dim * sizeof(Goldilocks::Element));
                nreads += dim;
            }
            else if (node["type"] == "trace")
            {
                // node data
                uint64_t segment = node["args"]["segment"];
                uint64_t col_offset = node["args"]["col_offset"];
                uint64_t row_offset = node["args"]["row_offset"];
                uint64_t ncols = traceWidths[segment];
                uint64_t dim = node["value"] == "base" ? 1 : degree;

                // asserts
                assert(segment >= 0 && segment < nSegments);
                assert(col_offset >= 0 && col_offset < ncols);
                assert(row_offset == 0 || row_offset == 1);

                // copy
                if (irow == N - 1 && row_offset == 1) // case last row and row_offset=1 (ciclic trace => trace value found in first row)
                {
                    memcpy(&nodesRes[k * 3], &trace[segment][col_offset], dim * sizeof(Goldilocks::Element));
                }
                else
                {
                    memcpy(&nodesRes[k * 3], &trace[segment][(irow + row_offset) * ncols + col_offset], dim * sizeof(Goldilocks::Element));
                }
                nreads += dim;
            }
            else if (node["type"] == "const")
            {
                // node data
                uint64_t value = node["args"]["value"];

                // asserts
                assert(node["value"] == "base");

                // copy
                nodesRes[k * degree] = Goldilocks::fromU64(value);
            }
            else
            {

                // node data
                uint64_t node1 = node["args"]["lhs"];
                uint64_t node2 = node["args"]["rhs"];
                uint64_t dim1 = nodes[node1]["value"] == "base" ? 1 : degree;
                uint64_t dim2 = nodes[node2]["value"] == "base" ? 1 : degree;
                uint64_t maxdim = std::max(dim1, dim2);
                Goldilocks::Element *input1 = &nodesRes[node1 * degree];
                Goldilocks::Element *input2 = &nodesRes[node2 * degree];
                Goldilocks::Element *output = &nodesRes[k * degree];

                // asserts
                assert(node1 * 3 >= 0 && node1 * 3 < nNodes * 3);
                assert(node2 * 3 >= 0 && node2 * 3 < nNodes * 3);

                if (node["type"] == "add")
                {
                    Goldilocks::add(output[0], input1[0], input2[0]);
                    nsums++;
                    for (uint64_t i = 1; i < maxdim; i++)
                    {
                        if (dim1 >= i && dim2 >= i)
                        {
                            Goldilocks::add(output[i], input1[i], input2[i]);
                            ++nsums;
                        }
                        else if (dim1 >= i)
                        {
                            output[i] = input1[i];
                        }
                        else if (dim2 >= i)
                        {
                            output[i] = input2[i];
                        }
                    }
                }
                else if (node["type"] == "sub")
                {
                    Goldilocks::sub(output[0], input1[0], input2[0]);
                    nsubs++;
                    for (uint64_t i = 1; i < maxdim; i++)
                    {
                        if (dim1 >= i && dim2 >= i)
                        {
                            Goldilocks::sub(output[i], input1[i], input2[i]);
                            ++nsubs;
                        }
                        else if (dim1 >= i)
                        {
                            output[i] = input1[i];
                        }
                        else if (dim2 >= i)
                        {
                            output[i] = -input2[i];
                        }
                    }
                }
                else if (node["type"] == "mul")
                {
                    if (dim1 == 1 || dim2 == 1)
                    {
                        Goldilocks::mul(output[0], input1[0], input2[0]);
                        nmuls++;
                        for (uint64_t i = 1; i < maxdim; i++)
                        {
                            if (dim1 == 1)
                            {
                                Goldilocks::mul(output[i], input1[0], input2[i]);
                            }
                            else
                            {
                                Goldilocks::mul(output[i], input1[i], input2[0]);
                            }
                            ++nmuls;
                        }
                    }
                    else
                    {
                        // Irreductible polynomial: x^3 - x - 1
                        // Karatsuba multiplication
                        Goldilocks::Element A = (input1[0] + input1[1]) * (input2[0] + input2[1]);
                        Goldilocks::Element B = (input1[0] + input1[2]) * (input2[0] + input2[2]);
                        Goldilocks::Element C = (input1[1] + input1[2]) * (input2[1] + input2[2]);
                        Goldilocks::Element D = input1[0] * input2[0];
                        Goldilocks::Element E = input1[1] * input2[1];
                        Goldilocks::Element F = input1[2] * input2[2];
                        Goldilocks::Element G = D - E;

                        output[0] = (C + G) - F;
                        output[1] = ((((A + C) - E) - E) - D);
                        output[2] = B - G;

                        nmuls += 6;
                        nsums += 8;
                        nsubs += 5;
                        assert(node["value"] == "ext");
                    }
                }
                else
                {
                    std::cout << "Node type " << node["type"] << std::endl;
                    throw runtime_error("Error: unknown node type");
                }
            }
        }

        // get the evaluations:
        uint64_t pos = 0;
        for (uint64_t i = 0; i < nExpressions; i++)
        {
            auto &expression = expressions[i];
            uint64_t node_id = expression["node_id"];
            uint64_t dim = nodes[node_id]["value"] == "base" ? 1 : degree;
            Goldilocks::Element *input = &nodesRes[node_id * degree];
            Goldilocks::Element *output = &expEvals[irow * expDim + pos];
            memcpy(output, input, dim * sizeof(Goldilocks::Element));
            pos += dim;
            nwrites += dim;
        }
        // Debug: print nodesRes
        /*if (irow == 0)
        {
            for (uint64_t i = 0; i < nNodes; ++i)
            {
                std::cout << "node " << i << ": " << nodesRes[3 * i].fe << " " << nodesRes[3 * i + 1].fe << " " << nodesRes[3 * i + 2].fe << std::endl;
            }
        }*/
    }

    //
    // Hash expEvals
    //
    Goldilocks::Element *hash = new Goldilocks::Element[CAPACITY];
    Goldilocks::Element *hashInput = &expEvals[0];
    PoseidonGoldilocks::linear_hash_seq(hash, hashInput, nExpressions * N);

    //
    //  Info
    //
    std::cout << endl;
    std::cout << "   > AST file:          " << fileName << endl;
    std::cout << "   > N:                 " << N << endl;
    std::cout << "   > trace widths:      [";
    for (uint64_t i = 0; i < nSegments; i++)
    {
        std::cout << traceWidths[i];
        if (i < nSegments - 1)
            std::cout << ", ";
    }
    std::cout << "]" << endl;
    std::cout << "   > num variables:     [";
    for (uint64_t i = 0; i < nVarGroups; i++)
    {
        std::cout << varGroupSizes[i];
        if (i < nVarGroups - 1)
            std::cout << ", ";
    }
    std::cout << "]" << endl;
    std::cout << "   > nNodes:            " << nNodes << endl;
    std::cout << "   > expEvals  HASH:    " << hash[0].fe << " " << hash[1].fe << " " << hash[2].fe << " " << hash[3].fe << endl;
    int ntotal = nsums + nsubs + nmuls + nreads + nwrites;
    std::cout << "   > num operations:    " << ntotal << endl
              << endl;
    std::cout << std::fixed << std::setprecision(1) << "      #sums:   " << nsums << " ( " << float(nsums) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "      #subs:   " << nsubs << " ( " << float(nsubs) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "      #muls:   " << nmuls << " ( " << float(nmuls) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "      #reads:  " << nreads << " ( " << float(nreads) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "      #writes: " << nwrites << " ( " << float(nwrites) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << endl;
    return 0;
}

//
// Helper functions
//
Goldilocks::Element fill_trace(uint64_t i, uint64_t j, uint64_t k)
{
    Goldilocks::Element e;
    e = Goldilocks::fromS64(i * 11 + j * 7 + k * 21);
    return e;
}

Goldilocks::Element fill_variable(uint64_t i, uint64_t j)
{
    Goldilocks::Element e;
    e = Goldilocks::fromS64(i * 31 + j * 101);
    return e;
}

void file2json(const string &fileName, json &j)
{
    j.clear();
    std::ifstream inputStream(fileName);
    if (!inputStream.good())
    {
        throw runtime_error("Error: not able to load file '" + fileName + "'");
    }
    try
    {
        inputStream >> j;
    }
    catch (exception &e)
    {
        throw runtime_error("Error parsing JSON from '" + fileName + "': " + e.what());
    }
    inputStream.close();
}