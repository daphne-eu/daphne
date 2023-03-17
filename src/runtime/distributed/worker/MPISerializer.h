
#ifndef SRC_RUNTIME_DISTRIBUTED_MPISERIALIZER_H
#define SRC_RUNTIME_DISTRIBUTED_MPISERIALIZER_H

#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <mpi.h>
class MPISerializer{
    public:

    static void serializeTask (void **taskToSend, size_t * length, distributed::Task * task)
    {
        *length = task->ByteSizeLong();
        *taskToSend  = (void *) malloc(*length * sizeof(unsigned char));
        task->SerializeToArray(*taskToSend,*length);
    }
    
    static void deserializeTask (distributed::Task * task, void * data, size_t length)
    {
        task->ParseFromArray(data,length);
    }
    
    template<class DT>
    static void serializeStructure(void ** dataToSend, DT *&mat, bool isScalar, size_t * length)
    {
       if(isScalar){
            serializeStructure(dataToSend, mat, isScalar, length,  0, 0, 0, 0);
       }
       else{
            serializeStructure(dataToSend, mat, isScalar, length, 0, mat->getNumRows(), 0, mat->getNumCols()); 
       }
   
    }
    
    template<class DT>
    static void serializeStructure(void ** dataToSend, DT *&mat, bool isScalar, size_t * length, size_t startRow, size_t rowCount, size_t startCol, size_t colCount){
        distributed::Data protoMsg;
        if (isScalar) {
            auto ptr = (double*)(&mat);
            double val = *ptr;
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false);
            auto protoVal = protoMsg.mutable_value();
            protoVal->set_f64(val);
        } 
        else 
        {
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix(), startRow, startRow+rowCount, startCol, startCol+colCount);
        }
        *length = protoMsg.ByteSizeLong();
        *dataToSend  = (void *) malloc(*length * sizeof(unsigned char));
        protoMsg.SerializeToArray(*dataToSend,*length);
    }
    
    static void deserializeStructure(distributed::Data * protoMsgData, void * data, size_t length){
        
        protoMsgData->ParseFromArray(data, length);
    }
};
#endif