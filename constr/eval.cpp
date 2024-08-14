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
Goldilocks::Element fill_variable(uint64_t i, uint64_t j, uint64_t k);

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

    //
    // Allocate memory for TRACE, VARIABLES, RESULTS and NODESRES
    //
    uint64_t nStages = ast["nStages"];
    std::vector<uint64_t> stagesCols = ast["stagesCols"].get<std::vector<uint64_t>>();
    vector<Goldilocks::Element *> trace(nStages);
    for (uint64_t i = 0; i < nStages; i++)
    {
        if (stagesCols[i] != 0)
        {
            trace[i] = new Goldilocks::Element[stagesCols[i] * N];
        }
        else
        {
            trace[i] = nullptr;
        }
    }
    uint64_t nVariables = ast["nVariables"];
    std::vector<pair<uint64_t, uint64_t>> varSizes = ast["variablesSize"].get<std::vector<pair<uint64_t, uint64_t>>>();
    vector<Goldilocks::Element *> variables(nVariables);

    for (uint64_t i = 0; i < nVariables; i++)
    {
        if (varSizes[i].first != 0)
        {
            variables[i] = new Goldilocks::Element[varSizes[i].first * varSizes[i].second];
        }
        else
        {
            variables[i] = nullptr;
        }
    }
    uint64_t nResCols = ast["nEvals"];
    Goldilocks::Element *evaluation = new Goldilocks::Element[nResCols * N];
    uint64_t nNodes = ast["nNodes"];
    Goldilocks::Element *nodesRes = new Goldilocks::Element[nNodes * 3]; // 3 is the size of the extended field

    //
    // Fill traces and variables and initialize evaluaitons and nodes to 0
    //
    for (uint64_t i = 0; i < nStages; i++)
    {
        for (uint64_t j = 0; j < N; ++j)
        {
            for (uint64_t k = 0; k < stagesCols[i]; ++k)
            {
                trace[i][j * stagesCols[i] + k] = fill_trace(i, j, k);
            }
        }
    }
    for (uint64_t i = 0; i < nVariables; i++)
    {
        for (uint64_t j = 0; j < varSizes[i].first; ++j)
        {
            for (uint64_t k = 0; k < varSizes[i].second; ++k)
            {
                variables[i][j * varSizes[i].second + k] = fill_variable(i, j, k);
            }
        }
    }
    for (uint64_t i = 0; i < N * nResCols; i++)
    {
        evaluation[i] = Goldilocks::zero();
    }

    //
    // Evaluate the expressions
    //
    uint64_t nsums = 0;
    uint64_t nsubs = 0;
    uint64_t nmuls = 0;
    uint64_t nreads = 0;
    uint64_t nwrites = 0;
    auto &nodes = ast["nodes"];

    // note: this loop is parallelizable!
    for (uint64_t irow = 0; irow < N; ++irow)
    {
        for (uint64_t i = 0; i < nNodes * 3; i++)
        {
            nodesRes[i] = Goldilocks::zero();
        }

        for (uint64_t k = 0; k < nNodes; k++)
        {
            auto &node = nodes[k];
            if (node["type"] == "VALUE")
            {
                if (node["name"] == "VARIABLE")
                {
                    uint64_t groupIdx = node["args"]["group_idx"];
                    uint64_t varIdx = node["args"]["var_idx"];

                    assert(groupIdx >= 0 && groupIdx < nVariables);
                    assert(varIdx >= 0 && varIdx < varSizes[groupIdx].first);
                    assert(varSizes[groupIdx].second == 1);
                    assert(node["resultType"] == "base");

                    nodesRes[k * 3] = variables[groupIdx][varIdx];
                    nodesRes[k * 3 + 1] = Goldilocks::zero();
                    nodesRes[k * 3 + 2] = Goldilocks::zero();
                    ++nreads;
                }
                else if (node["name"] == "VARIABLE3")
                {
                    uint64_t groupIdx = node["args"]["group_idx"];
                    uint64_t varIdx = node["args"]["var_idx"];

                    assert(groupIdx >= 0 && groupIdx < nVariables);
                    assert(varIdx >= 0 && varIdx < varSizes[groupIdx].first);
                    assert(varSizes[groupIdx].second == 3);
                    assert(node["resultType"] == "extension");

                    nodesRes[k * 3] = variables[groupIdx][3 * varIdx];
                    nodesRes[k * 3 + 1] = variables[groupIdx][3 * varIdx + 1];
                    nodesRes[k * 3 + 2] = variables[groupIdx][3 * varIdx + 2];
                    nreads += 3;
                }
                else if (node["name"] == "TRACE")
                {
                    uint64_t stageIdx = node["args"]["stage_idx"];
                    uint64_t colIdx = node["args"]["column_idx"];
                    uint64_t row_offset = node["args"]["row_offset"];
                    uint64_t ncols = stagesCols[stageIdx];

                    assert(stageIdx >= 0 && stageIdx < nStages);
                    assert(colIdx >= 0 && colIdx < ncols);
                    assert(row_offset == 0 || row_offset == 1);
                    assert(node["resultType"] == "base");

                    if (irow == N - 1 && row_offset == 1)
                    {
                        nodesRes[k * 3] = trace[stageIdx][colIdx];
                    }
                    else
                    {
                        nodesRes[k * 3] = trace[stageIdx][(irow + row_offset) * ncols + colIdx];
                    }
                    nodesRes[k * 3 + 1] = Goldilocks::zero();
                    nodesRes[k * 3 + 2] = Goldilocks::zero();
                    ++nreads;
                }
                else if (node["name"] == "TRACE3")
                {
                    uint64_t stageIdx = node["args"]["stage_idx"];
                    uint64_t colIdx = node["args"]["column_idx"];
                    uint64_t row_offset = node["args"]["row_offset"];
                    uint64_t ncols = stagesCols[stageIdx];

                    assert(stageIdx >= 0 && stageIdx < nStages);
                    assert(colIdx >= 0 && colIdx < ncols);
                    assert(row_offset == 0 || row_offset == 1);
                    assert(node["resultType"] == "extension");

                    if (irow == N - 1 && row_offset == 1)
                    {
                        nodesRes[k * 3] = trace[stageIdx][colIdx];
                        nodesRes[k * 3 + 1] = trace[stageIdx][colIdx + 1];
                        nodesRes[k * 3 + 2] = trace[stageIdx][colIdx + 2];
                    }
                    else
                    {
                        nodesRes[k * 3] = trace[stageIdx][(irow + row_offset) * ncols + colIdx];
                        nodesRes[k * 3 + 1] = trace[stageIdx][(irow + row_offset) * ncols + colIdx + 1];
                        nodesRes[k * 3 + 2] = trace[stageIdx][(irow + row_offset) * ncols + colIdx + 2];
                    }
                    nreads += 3;
                }
                else if (node["name"] == "EVAL")
                {
                    uint64_t nodeIdx = node["args"]["node_out"];
                    uint64_t destIdx = node["args"]["dest_col"];

                    assert(nodeIdx >= 0 && nodeIdx < nNodes);
                    assert(destIdx >= 0 && destIdx < nResCols);
                    assert(node["resultType"] == "base");

                    evaluation[irow * nResCols + destIdx] = nodesRes[nodeIdx * 3];
                    ++nwrites;
                }
                else if (node["name"] == "EVAL3")
                {
                    uint64_t nodeIdx = node["args"]["node_out"];
                    uint64_t destIdx = node["args"]["dest_col"];

                    assert(nodeIdx >= 0 && nodeIdx < nNodes);
                    assert(destIdx >= 0 && destIdx + 2 < nResCols);
                    assert(node["resultType"] == "extension");

                    evaluation[irow * nResCols + destIdx] = nodesRes[nodeIdx * 3];
                    evaluation[irow * nResCols + destIdx + 1] = nodesRes[nodeIdx * 3 + 1];
                    evaluation[irow * nResCols + destIdx + 2] = nodesRes[nodeIdx * 3 + 2];
                    nwrites += 3;
                }
                else
                {
                    throw runtime_error("Error: unknown VALUE type");
                }
            }
            else
            {
                assert(node["type"] == "OP");

                uint64_t node1 = node["args"]["node1"];
                uint64_t node2 = node["args"]["node2"];
                assert(node1 * 3 >= 0 && node1 * 3 < nNodes * 3);
                assert(node2 * 3 >= 0 && node2 * 3 < nNodes * 3);
                Goldilocks::Element *input1 = &nodesRes[node1 * 3];
                Goldilocks::Element *input2 = &nodesRes[node2 * 3];
                Goldilocks::Element *output = &nodesRes[k * 3];

                if (node["name"] == "ADD")
                {
                    Goldilocks::add(output[0], input1[0], input2[0]);
                    nsums++;
                    assert(node["resultType"] == "base");
                }
                else if (node["name"] == "ADD31")
                {
                    Goldilocks::add(output[0], input1[0], input2[0]);
                    output[1] = input1[1];
                    output[2] = input1[2];
                    nsums++;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "ADD33")
                {
                    Goldilocks::add(output[0], input1[0], input2[0]);
                    Goldilocks::add(output[1], input1[1], input2[1]);
                    Goldilocks::add(output[2], input1[2], input2[2]);
                    nsums += 3;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "SUB")
                {
                    Goldilocks::sub(output[0], input1[0], input2[0]);
                    nsubs++;
                    assert(node["resultType"] == "base");
                }
                else if (node["name"] == "SUB31")
                {
                    Goldilocks::sub(output[0], input1[0], input2[0]);
                    output[1] = input1[1];
                    output[2] = input1[2];
                    nsubs++;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "SUB13")
                {
                    Goldilocks::sub(output[0], input1[0], input2[0]);
                    output[1] = -input2[1];
                    output[2] = -input2[2];
                    nsubs += 3;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "SUB33")
                {
                    Goldilocks::sub(output[0], input1[0], input2[0]);
                    Goldilocks::sub(output[1], input1[1], input2[1]);
                    Goldilocks::sub(output[2], input1[2], input2[2]);
                    nsubs += 3;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "MULT")
                {
                    Goldilocks::mul(output[0], input1[0], input2[0]);
                    assert(node["resultType"] == "base");
                    nmuls++;
                }
                else if (node["name"] == "MULT31")
                {
                    Goldilocks::mul(output[0], input1[0], input2[0]);
                    Goldilocks::mul(output[1], input1[1], input2[0]);
                    Goldilocks::mul(output[2], input1[2], input2[0]);
                    nmuls += 3;
                    assert(node["resultType"] == "extension");
                }
                else if (node["name"] == "MULT33")
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
                    assert(node["resultType"] == "extension");
                }
                else
                {
                    std::cout << "operation " << node["name"] << std::endl;
                    throw runtime_error("Error: unknown OP type");
                }
            }
        }
    }
    //
    // Hash evaluation
    //

    /*for (uint64_t i = 0; i < nNodes; ++i)
    {
        std::cout << "node " << i << ": " << nodesRes[3 * i].fe << " " << nodesRes[3 * i + 1].fe << " " << nodesRes[3 * i + 1].fe << std::endl;
    }*/
    Goldilocks::Element *hash = new Goldilocks::Element[CAPACITY];
    Goldilocks::Element *hashInput = &evaluation[0];
    PoseidonGoldilocks::linear_hash_seq(hash, hashInput, nResCols * N);

    //
    //  Info
    //
    std::cout << endl;
    std::cout << "AST file: " << fileName << endl;
    std::cout << "N = " << N << endl;
    std::cout << "nStages = " << ast["nStages"] << endl;
    std::cout << "stagesCols = [";
    for (uint64_t i = 0; i < nStages; i++)
    {
        std::cout << stagesCols[i];
        if (i < nStages - 1)
            std::cout << ", ";
    }
    std::cout << "]" << endl;
    std::cout << "nVariables = " << nVariables << endl;
    std::cout << "variablesSize = [";
    for (uint64_t i = 0; i < nVariables; i++)
    {
        std::cout << "(" << varSizes[i].first << ", " << varSizes[i].second << ")";
        if (i < nVariables - 1)
            std::cout << ", ";
    }
    std::cout << "]" << endl;
    std::cout << "nNodes = " << nNodes << endl;
    std::cout << "evaluations HASH = " << hash[0].fe << " " << hash[1].fe << " " << hash[2].fe << " " << hash[3].fe << endl;
    int ntotal = nsums + nsubs + nmuls + nreads + nwrites;
    std::cout << std::fixed << std::setprecision(1) << "#sums: " << nsums << " ( " << float(nsums) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "#subs: " << nsubs << " ( " << float(nsubs) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "#muls: " << nmuls << " ( " << float(nmuls) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "#reads: " << nreads << " ( " << float(nreads) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << std::fixed << std::setprecision(1) << "#writes: " << nwrites << " ( " << float(nwrites) / float(ntotal) * 100.0 << "%)" << endl;
    std::cout << endl;

    return 0;
}

Goldilocks::Element fill_trace(uint64_t i, uint64_t j, uint64_t k)
{
    Goldilocks::Element e;
    e = Goldilocks::fromS64(i * 3 + j * 29 + k * 47);
    return e;
}

Goldilocks::Element fill_variable(uint64_t i, uint64_t j, uint64_t k)
{
    Goldilocks::Element e;
    e = Goldilocks::fromS64(i * 67 + j * 11 + k * 41);
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