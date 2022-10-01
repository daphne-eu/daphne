
#ifndef SRC_RUNTIME_DISTRIBUTED_MPISERIALIZER_H
#define SRC_RUNTIME_DISTRIBUTED_MPISERIALIZER_H

#include <runtime/distributed/proto/ProtoDataConverter.h>
#include <mpi.h>
template<class DT>
class MPISerializer{
    public:
    static void serialize(void ** dataToSend, DT *&mat, bool isScalar, size_t * length)
    {
       if(isScalar){
            serialize(dataToSend, mat, isScalar, length,  0, 0, 0, 0);
       }
       else{
            serialize(dataToSend, mat, isScalar, length, 0, mat->getNumRows(), 0, mat->getNumCols()); 
       }
   
    }
    static void serialize(void ** dataToSend, DT *&mat, bool isScalar, size_t * length, size_t startRow, size_t rowCount, size_t startCol, size_t colCount){
        distributed::Data protoMsg;
        if (isScalar) {
            auto ptr = (double*)(&mat);
            double* val = ptr;
            mat = DataObjectFactory::create<DenseMatrix<double>>(0, 0, false); 
            auto protoVal = protoMsg.mutable_value();
            protoVal->set_f64(*val);
        } 
        else 
        {
            auto denseMat = dynamic_cast<const DenseMatrix<double>*>(mat);
            if (!denseMat){
                throw std::runtime_error("Distribute grpc only supports DenseMatrix<double> for now");
            }
            ProtoDataConverter<DenseMatrix<double>>::convertToProto(denseMat, protoMsg.mutable_matrix(), startRow, rowCount, startCol, colCount);
        }
        *length = protoMsg.ByteSizeLong();
        *dataToSend  = (void *) malloc(*length * sizeof(unsigned char));
        protoMsg.SerializeToArray(*dataToSend,*length);
    }
    static DT* deserialize(void * data, size_t length){
        distributed::Data protoMsg;
        protoMsg.ParseFromArray(data,length);
        const distributed::Matrix& mat = protoMsg.matrix();
        
        auto temp= DataObjectFactory::create<DT>(protoMsg.mutable_matrix()->num_rows(), protoMsg.mutable_matrix()->num_cols(), false);
        DT *res =  dynamic_cast<DT *>(temp);
        //ProtoDataConverter<DenseMatrix<double>>::test();
        //ProtoDataConverter<DenseMatrix<double>>::test1(mat);
        //ProtoDataConverter<DenseMatrix<double>>::test2(res);
       // ProtoDataConverter<DenseMatrix<double>>::convertFromProto<>(mat, res);
        ProtoDataConverter<DT>::convertFromProto(mat, res);
        return res;
    }
};
#endif