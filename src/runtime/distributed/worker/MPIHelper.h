#ifndef SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H
#define SRC_RUNTIME_DISTRIBUTED_MPIHELPER_H

#include <mpi.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/worker/WorkerImpl.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <runtime/local/datastructures/AllocationDescriptorMPI.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#include <vector>

#define COORDINATOR 0

enum TypesOfMessages{
    BROADCAST, STREAM_INIT, STREAM_COMPLETE, DATASIZE, DATA, DATAACK, MLIRSIZE, TRANSFER, TRANSFERSIZE, MLIR, INPUTKEYS, COMPUTERESULT, OUTPUT, OUTPUTKEY,  DETACH
};
enum WorkerStatus{
    LISTENING=0, DETACHED, TERMINATED
};

class MPIHelper
{
public:
    using StoredInfo = WorkerImpl::StoredInfo;
    struct Task {
    private:
        struct Header {
            size_t mlir_code_len;
            size_t num_inputs;
        } __attribute__((__packed__));
    public:
        std::string mlir_code;
        std::vector<WorkerImpl::StoredInfo> inputs;

        size_t sizeInBytes() { 
            size_t len = 0;
            len += sizeof(Header);
            len += mlir_code.size();
            for (auto &inp : inputs){
                len += sizeof(size_t); // strlen
                len += inp.identifier.size();
                len += sizeof(size_t) * 2; // numRows + numCols
            }
            return len;
        }
        void serialize(std::vector<char> &buffer){
            Header h;
            h.mlir_code_len = mlir_code.size();
            h.num_inputs = inputs.size();
            
            buffer.resize(this->sizeInBytes());

            auto bufIdx = buffer.begin();
            std::copy(reinterpret_cast<char*>(&h), reinterpret_cast<char*>(&h) + sizeof(h), bufIdx);
            bufIdx += sizeof(h);

            std::copy(mlir_code.begin(), mlir_code.end(), bufIdx);
            bufIdx += mlir_code.size();
                        
            for (auto &inp : inputs){
                size_t strLen = inp.identifier.size();
                std::copy(reinterpret_cast<char*>(&strLen), reinterpret_cast<char*>(&strLen) + sizeof(strLen), bufIdx);
                bufIdx += sizeof(strLen);
                std::copy(inp.identifier.data(), inp.identifier.data() + strLen, bufIdx);
                bufIdx += strLen;
                std::copy(reinterpret_cast<char*>(&inp.numRows), reinterpret_cast<char*>(&inp.numRows) + sizeof(inp.numRows), bufIdx);
                bufIdx += sizeof(inp.numRows);
                std::copy(reinterpret_cast<char*>(&inp.numCols), reinterpret_cast<char*>(&inp.numCols) + sizeof(inp.numCols), bufIdx);
                bufIdx += sizeof(inp.numCols);
            }
        }
        void deserialize(const std::vector<char> &buffer){
            size_t mlir_code_len = (size_t)((const Header*)buffer.data())->mlir_code_len;
            size_t num_inputs = (size_t)((const Header*)buffer.data())->num_inputs;

            auto bufIdx = buffer.begin();
            bufIdx += sizeof(Header);

            this->mlir_code.resize(mlir_code_len);
            std::copy(bufIdx, bufIdx + mlir_code_len, mlir_code.begin());
            bufIdx += mlir_code_len;
            
            this->inputs.resize(num_inputs);
            
            for (auto &inp : inputs){
                size_t strLen;
                std::copy(bufIdx, bufIdx + sizeof(strLen), reinterpret_cast<char*>(&strLen));
                bufIdx += sizeof(strLen);
                inp.identifier.resize(strLen);
                std::copy(bufIdx, bufIdx + strLen, inp.identifier.data());
                bufIdx += strLen;
                std::copy(bufIdx, bufIdx + sizeof(inp.numRows), reinterpret_cast<char*>(&inp.numRows));
                bufIdx += sizeof(inp.numRows);
                std::copy(bufIdx, bufIdx + sizeof(inp.numCols), reinterpret_cast<char*>(&inp.numCols));
                bufIdx += sizeof(inp.numCols);
            }
        }
    };

    static int getCommSize()
    {
        int worldSize;
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        return worldSize;
    }
    

    static WorkerImpl::StoredInfo constructStoredInfo(std::string input)
    {
        WorkerImpl::StoredInfo info;
        std::stringstream s_stream(input);
        std::vector<std::string> results;
        while (s_stream.good())
        {
            std::string substr;
            getline(s_stream, substr, ','); // get first string delimited by comma
            results.push_back(substr);
        }
        info.identifier = results.at(0);
        sscanf(results.at(1).c_str(), "%zu", &info.numRows);
        sscanf(results.at(2).c_str(), "%zu", &info.numCols);
        return info;
    }
    static std::vector<WorkerImpl::StoredInfo> constructStoredInfoVector(std::vector<char> &buffer)
    {
        std::vector<WorkerImpl::StoredInfo> vecInfo;
        std::string str(buffer.begin(), buffer.end());
        std::stringstream s_stream(str);
        std::string substr;
        while (getline(s_stream, substr, ':'))
        {
            vecInfo.push_back(constructStoredInfo(substr));
        }
        return vecInfo;
    }

    static std::vector<char> getComputeResults(int rank)
    {
        size_t resultsLen = 0;
        std::vector<char> buffer;
        getMessageFrom(rank, COMPUTERESULT, MPI_UNSIGNED_CHAR, buffer, &resultsLen);
        return buffer;
    }

    static WorkerImpl::StoredInfo getDataAcknowledgement(int *rank)
    {
        std::vector<char> dataAcknowledgement;
        size_t len;
        getMessage(rank, DATAACK, MPI_CHAR, dataAcknowledgement, &len);
        std::string incomeAck = std::string(dataAcknowledgement.data());
        StoredInfo info = constructStoredInfo(incomeAck);        
        return info;
    }

    static void broadcastData(size_t messageLength, void *data)
    {
        int worldSize = getCommSize();
        int message = messageLength;
        for (int rank = 0; rank < worldSize; rank++)
        {
            if (rank == COORDINATOR)
                continue;
            MPI_Send(&message, 1, MPI_INT, rank, BROADCAST, MPI_COMM_WORLD);
        }
        MPI_Bcast(data, message, MPI_UNSIGNED_CHAR, COORDINATOR, MPI_COMM_WORLD);
    }

    static void initiateStreaming(int rank, size_t chunksize)
    {
        MPI_Send(&chunksize, 1, MPI_INT, rank, STREAM_INIT, MPI_COMM_WORLD);
    }
    static void sendData(size_t messageLength, void *data, int rank)
    {
        sendWithTag(DATA, messageLength, data, rank);
    }

    static void sendTask(size_t messageLength, void *data, int rank)
    {
        sendWithTag(MLIR, messageLength, data, rank);
    }

    static void displayDataStructure(Structure *inputStruct, std::string dataToDisplay)
    {
        DenseMatrix<double> *res = dynamic_cast<DenseMatrix<double> *>(inputStruct);
        double *allValues = res->getValues();
        for (size_t r = 0; r < res->getNumRows(); r++)
        {
            for (size_t c = 0; c < res->getNumCols(); c++)
            {
                dataToDisplay += std::to_string(allValues[c]) + " , ";
            }
            dataToDisplay += "\n";
            allValues += res->getRowSkip();
        }
        // std::cout<<dataToDisplay<<std::endl;
    }

    static void requestData(const int& rank, const StoredInfo& info)
    {
        int len = info.toString().length();
        len++;
        MPI_Send(&len, 1, MPI_INT, rank, TRANSFERSIZE, MPI_COMM_WORLD);
        char message[len];
        std::strcpy(message, info.toString().c_str());
        message[len - 1] = '\0';
        MPI_Send(message, len, MPI_CHAR, rank, TRANSFER, MPI_COMM_WORLD);
    }


    static void getMessage(int *rank, int tag, MPI_Datatype type, std::vector<char> &data, size_t *len)
    {
        int size;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, type, &size);
        *rank = status.MPI_SOURCE;

        data.resize(size * sizeof(char));
        MPI_Recv(data.data(), size, type, status.MPI_SOURCE, tag, MPI_COMM_WORLD, &status);
        *len = size;
    }

    static void getMessageFrom(int rank, int tag, MPI_Datatype type, std::vector<char> &data, size_t *len)
    {
        int size;
        MPI_Status status;
        MPI_Probe(rank, tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, type, &size);

        data.resize(size * sizeof(char));
        MPI_Recv(data.data(), size, type, rank, tag, MPI_COMM_WORLD, &status);
        *len = size;
    }

    static void sendWithTag(TypesOfMessages tag, size_t messageLength, void *data, int rank)
    {
        if (rank == COORDINATOR)
            return;
        int message = messageLength;
        int sizeTag = -1, dataTag = -1;
        // std::cout<<"message size is "<< message << " tag "<< tag <<std::endl;
        switch (tag)
        {
        case DATA:
            sizeTag = DATASIZE;
            dataTag = DATA;
            break;
        case MLIR:
            sizeTag = MLIRSIZE;
            dataTag = MLIR;
        default:
            break;
        }
        // std::cout<<"message size is "<< message << " tag "<< sizeTag <<std::endl;
        MPI_Send(&message, 1, MPI_INT, rank, sizeTag, MPI_COMM_WORLD);
        MPI_Send(data, message, MPI_UNSIGNED_CHAR, rank, dataTag, MPI_COMM_WORLD);
    }
};

#endif