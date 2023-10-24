#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_DISTRIBUTEDALLOCATIONHELPER_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_DISTRIBUTEDALLOCATIONHELPER_H

class DistributedIndex
{
    public:
        DistributedIndex() : row_(0), col_(0)
        {}
        DistributedIndex(size_t row, size_t col) : row_(row), col_(col)
        {}

        size_t getRow() const
        {
            return row_;
        }
        size_t getCol() const
        {
            return col_;
        }

        bool operator<(const DistributedIndex rhs) const
        {
            if (row_ < rhs.row_)
                return true;
            else if (row_ == rhs.row_)
                return col_ < rhs.col_;
            return false;
        }
    private:
        size_t row_;
        size_t col_;
};


struct DistributedData
{
    std::string identifier;
    size_t numRows, numCols;
    mlir::daphne::VectorCombine vectorCombine;
    bool isPlacedAtWorker = false;
    DistributedIndex ix;

};

#endif // SRC_RUNTIME_LOCAL_DATASTRUCTURES_DISTRIBUTEDALLOCATIONHELPER_H