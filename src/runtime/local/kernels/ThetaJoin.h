/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef SRC_RUNTIME_LOCAL_KERNELS_THETAJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_THETAJOIN_H

#include <util/DeduceType.h>

enum CompareOperation {
    Equal,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    NotEqual
};


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
class ThetaJoin {
  public:
    static void apply(DTRes*& res, const DTLhs* lhs, const DTRhs* rhs, const char** lhsOn, size_t numLhsOn,
                      const char** rhsOn, size_t numRhsOn, CompareOperation* cmp, size_t numCmp) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTLhs, class DTRhs>
void thetaJoin(DTRes*& res, const DTLhs* lhs, const DTRhs* rhs, const char** lhsOn, size_t numLhsOn,
               const char** rhsOn, size_t numRhsOn, CompareOperation* cmp, size_t numCmp){
    ThetaJoin<DTRes, DTLhs, DTRhs>::apply(res, lhs, rhs, lhsOn, numLhsOn, rhsOn, numRhsOn, cmp, numCmp);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame, Frame -> Frame
// ----------------------------------------------------------------------------

template<>
class ThetaJoin<Frame, Frame, Frame> {
    /**
     * Stores column indices and compare operation of one equation.
     */
    struct Equation {
        size_t lhsColumnIndex;
        size_t rhsColumnIndex;
        CompareOperation cmp;
        
        Equation(size_t lhsColumnIndex, size_t rhsColumnIndex, CompareOperation cmp)
        : lhsColumnIndex(lhsColumnIndex), rhsColumnIndex(rhsColumnIndex), cmp(cmp){
            // do nothing
        }
    };

    /**
     * @brief Convenience structure to store both relations and give easy access to and creation of meta data.
     */
    struct Container {
        const Frame * const lhs = nullptr;
        const Frame * const rhs = nullptr;
        
        const ValueTypeCode * lhsSchema = nullptr;
        const ValueTypeCode * rhsSchema = nullptr;
        std::vector<Equation> equations;
    
        void addEquation(const char* lhsOn, const char* rhsOn, CompareOperation cmp){
            equations.emplace_back(lhs->getColumnIdx(lhsOn), rhs->getColumnIdx(rhsOn), cmp);
        }
        
        Container(const Frame * lhs, const Frame * rhs, const char ** lhsOn, const char ** rhsOn,
                  CompareOperation * cmp, uint64_t numCmp)
        : lhs(lhs), rhs(rhs), lhsSchema(lhs->getSchema()), rhsSchema(rhs->getSchema()){
            for(size_t i = 0; i < numCmp; ++i){
                addEquation(lhsOn[i], rhsOn[i], cmp[i]);
            }
        }
        
        /**
         * Create Frame schema for joining both relations.
         * @return schema
         */
        [[nodiscard]] ValueTypeCode* createResultSchema() const{
            size_t lhsCols = lhs->getNumCols();
            size_t rhsCols = rhs->getNumCols();
            
            auto resSchema = new ValueTypeCode[lhsCols + rhsCols];
            for(uint64_t i = 0; i < lhsCols; ++i){
                resSchema[i] = lhsSchema[i];
            }
            for(uint64_t i = 0; i < rhsCols; ++i){
                resSchema[i + lhsCols] = rhsSchema[i];
            }
            return resSchema;
        }
        
        /**
         * Create labels for joining both relations
         * @return
         */
        [[nodiscard]] std::string* createResultLabels() const{
            size_t lhsCols = lhs->getNumCols();
            size_t rhsCols = rhs->getNumCols();
            auto lhsLabels = lhs->getLabels();
            auto rhsLabels = rhs->getLabels();
            
            auto resLabels = new std::string[lhsCols + rhsCols];
            for(uint64_t i = 0; i < lhsCols; ++i){
                resLabels[i] = lhsLabels[i];
            }
            for(uint64_t i = 0; i < rhsCols; ++i){
                resLabels[i + lhsCols] = rhsLabels[i];
            }
            return resLabels;
        }
        
        /**
         * Get number of columns after joining both relations
         * @return number of columns
         */
        [[nodiscard]] size_t getResultNumCols() const{
            return lhs->getNumCols() + rhs->getNumCols();
        }
        
        /**
         * Get the ValueTypeCode for lhs column of eq_index-th equation.
         * @param eq_index index of equation
         * @return
         */
        ValueTypeCode getVTLhs(uint64_t eq_index){
            return lhsSchema[equations.at(eq_index).lhsColumnIndex];
        }
        /**
         * Get the ValueTypeCode for lhs column of eq_index-th equation.
         * @param eq_index index of equation
         * @return
         */
        ValueTypeCode getVTRhs(uint64_t eq_index){
            return rhsSchema[equations.at(eq_index).rhsColumnIndex];
        }
    };
    
    /**
     * Structure to store and manage the position lists.
     * Only usable for Conjunction of equations (Disjunction is not supported (yet))!
     */
    class ResultContainer {
        using posType = uint64_t;
        
        DenseMatrix<posType> * positions = nullptr;
        
        uint64_t readOffset = 0;
        uint64_t writeOffset = 0;
        uint64_t size_ = 0;
        uint64_t maxSize = 0;
        
      public:
        explicit ResultContainer(uint64_t maxNumRows)
        : positions(DataObjectFactory::create<DenseMatrix<posType>>(maxNumRows, 2, false)),
          maxSize(maxNumRows)
        {
            resetCursor();
        }
        
        ~ResultContainer(){
            DataObjectFactory::destroy(positions);
        }
        
        void resetCursor(){
            readOffset = 0;
            writeOffset = 0;
        }
        
        void addPosPair(uint64_t lhsPos_, uint64_t rhsPos_){
            /// lhs
            positions->set(writeOffset, 0, lhsPos_);
            /// rhs
            positions->set(writeOffset, 1, rhsPos_);
            ++writeOffset;
        }
        
        [[nodiscard]] std::tuple<posType, posType> readNext(){
            auto res = std::make_tuple(
              /// lhs
              positions->get(readOffset, 0),
              /// rhs
              positions->get(readOffset, 1));
            ++readOffset;
            return std::move(res);
        }
        
        void finalize(){
            size_ = writeOffset;
            resetCursor();
        }
        
        [[nodiscard]] uint64_t size() const {
            return size_;
        }
        
        [[nodiscard]] const DenseMatrix<posType> * getPositions() const {
            return positions;
        }
    };

    

//    template< template < typename ...> typename TExec, typename ... TArgs>
//    struct DeduceValueType {
//        template< typename ... TPar >
//        static void apply(ValueTypeCode vtc, TPar ... args) {
//            switch (vtc) {
//                case ValueTypeCode::SI8:
//                    TExec<TArgs..., int8_t>::apply(args...);
//                    return;
//                case ValueTypeCode::SI32:
//                    TExec<TArgs..., int32_t>::apply(args...);
//                    return;
//                case ValueTypeCode::SI64:
//                    TExec<TArgs..., int64_t>::apply(args...);
//                    return;
//                case ValueTypeCode::UI8:
//                    TExec<TArgs..., uint8_t>::apply(args...);
//                    return;
//                case ValueTypeCode::UI32:
//                    TExec<TArgs..., uint32_t>::apply(args...);
//                    return;
//                case ValueTypeCode::UI64:
//                    TExec<TArgs..., uint64_t>::apply(args...);
//                    return;
//                case ValueTypeCode::F32:
//                    TExec<TArgs..., float>::apply(args...);
//                    return;
//                case ValueTypeCode::F64:
//                    TExec<TArgs..., double>::apply(args...);
//                    return;
//                default:
//                    throw std::runtime_error(
//                      "deduceType: Unknown value type '" + std::to_string(static_cast<uint64_t>(vtc)) + "' found."
//                    );
//            }
//        }
//    };
    
//    template < template < template < typename ... > typename, typename ... > typename TExec, template < typename... > typename TSubExec, typename ... TArgs>
//    struct DeduceValueType <TExec<TSubExec, TArgs ...>, TArgs...> {};
    
    template< typename VTCol >
    struct WriteColumn {
        static void apply(Frame *& out, const Container& container, uint64_t inColIdx, uint64_t outColIdx,
                          bool isLhs, ResultContainer const * positions){
            if(!out){
                throw std::runtime_error("Result Frame not allocated!");
            }
            const Frame * in = isLhs ? container.lhs : container.rhs;
            
//            VTCol const * inData = in->template getColumn<VTCol>(inColIdx)->getValues();
            auto * inData = reinterpret_cast<VTCol const *>(in->getColumnRaw(inColIdx));
            auto * outData = reinterpret_cast<VTCol *>(out->getColumnRaw(outColIdx));
            
            for(uint64_t i = 0; i < positions->size(); ++i){
                outData[i] = inData[positions->getPositions()->get(i, isLhs ? 0 : 1)];
            }
        }
    };
    
//    template< typename VTLhs, typename VTRhs >
//    static bool compare(
//      /// Container, containing Frames
//      Container& state,
//      /// current loop pointer
//      size_t lhsIndex, size_t rhsIndex,
//      /// traverse depth
//      size_t depth)
//    {
//        Equation& eq = state.equations.at(depth);
//        VTLhs const * lhsData = state.lhs->template getColumn<VTLhs>(eq.lhsColumnIndex)->getValues();
//        VTRhs const * rhsData = state.rhs->template getColumn<VTRhs>(eq.rhsColumnIndex)->getValues();
//        bool result;
//        switch( eq.cmp ){
//            case Equal:
//                result = lhsData[lhsIndex] == rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " == " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//            case LessThan:
//                result = lhsData[lhsIndex] < rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " < " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//            case LessEqual:
//                result = lhsData[lhsIndex] <= rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " <= " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//            case GreaterThan:
//                result = lhsData[lhsIndex] > rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " > " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//            case GreaterEqual:
//                result = lhsData[lhsIndex] >= rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " >= " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//            case NotEqual:
//                result = lhsData[lhsIndex] != rhsData[rhsIndex];
//                cout << lhsData[lhsIndex] << " <> " << rhsData[rhsIndex] << " is " << result << endl;
//                return result;
//        }
//        return true;
//    }
//
//    template<typename VTLhs>
//    static bool compareRhs(
//      /// Container, containing Frames
//      Container& state,
//      /// current loop pointer
//      size_t lhsIndex, size_t rhsIndex,
//      /// traverse depth
//      size_t depth)
//    {
//        /// determine correct value type of the right hand side
//        switch ( state.rhsSchema[state.equations.at(depth).rhsColumnIndex] ){
//            case ValueTypeCode::SI8:
//                return compare<VTLhs, int8_t  >(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::SI32:
//                return compare<VTLhs, int32_t >(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::SI64:
//                return compare<VTLhs, int64_t >(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI8:
//                return compare<VTLhs, uint8_t >(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI32:
//                return compare<VTLhs, uint32_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI64:
//                return compare<VTLhs, uint64_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::F32:
//                return compare<VTLhs, float   >(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::F64:
//                return compare<VTLhs, double  >(state, lhsIndex, rhsIndex, depth);
//            default:
//                throw std::runtime_error("compareRhs: Unknown value type '" + std::to_string((uint64_t)state.rhsSchema[depth]) + "' found.");
//      }
//    }
//
//    static bool compareLhs(
//      /// Container, containing Frames
//      Container& state,
//      /// current loop pointer
//      size_t lhsIndex, size_t rhsIndex,
//      /// traverse depth
//      size_t depth)
//    {
//        /// determine correct value type of the left hand side
//        switch ( state.lhsSchema[state.equations.at(depth).lhsColumnIndex] ){
//            case ValueTypeCode::SI8:
//                return compareRhs<int8_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::SI32:
//                return compareRhs<int32_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::SI64:
//                return compareRhs<int64_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI8:
//                return compareRhs<uint8_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI32:
//                return compareRhs<uint32_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::UI64:
//                return compareRhs<uint64_t>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::F32:
//                return compareRhs<float>(state, lhsIndex, rhsIndex, depth);
//            case ValueTypeCode::F64:
//                return compareRhs<double>(state, lhsIndex, rhsIndex, depth);
//            default:
//                throw std::runtime_error("compareLhs: Unknown value type '" + std::to_string((uint64_t)state.rhsSchema[depth]) + "' found.");
//        }
//    }
//
//
//    static bool traverse(
//      /// Container, containing Frames
//      Container& state,
//      /// current loop pointer
//      size_t lhsIndex, size_t rhsIndex,
//      /// traverse depth
//      size_t depth = 0UL
//    )
//    {
//        /// compare both sides
//        bool hit = compareLhs(state, lhsIndex, rhsIndex, depth);
//        /// if single condition not met, whole condition is false (conjunction)
//        if(!hit){
//            return false;
//        }
//        /// hit == true // check if this was the last condition to check
//        if((depth+1) >= state.equations.size()){
//            return true;
//        } else {
//            /// go to next condition
//            return traverse(state, lhsIndex, rhsIndex, depth+1);
//        }
//    }
    
    
    
    /**
     * @brief Compares two values of arbitrary type following the encoded operation.
     *
     * @tparam VTLhs type of left hand side value
     * @tparam VTRhs type of right hand side value
     * @param lhs left hand side value
     * @param rhs right hand side value
     * @param cmp compare operation to use
     * @return boolean representation of the compare operation
     */
    template<typename VTLhs, typename VTRhs>
    static bool compareValues(VTLhs lhs, VTRhs rhs, CompareOperation cmp){
//        if constexpr ( !std::is_same_v<VTLhs, VTRhs>){
//            throw std::runtime_error("Value types do not match!");
//        } else { ...
        if constexpr (std::is_same_v<VTLhs, VTRhs>){
            switch ( cmp ){
                case Equal:
                    return lhs == rhs;
                case LessThan:
                    return lhs < rhs;
                case LessEqual:
                    return lhs <= rhs;
                case GreaterThan:
                    return lhs > rhs;
                case GreaterEqual:
                    return lhs >= rhs;
                case NotEqual:
                    return lhs != rhs;
            }
        } else {
            switch (cmp) {
                case Equal:
                    return lhs == static_cast<VTLhs>(rhs);
                case LessThan:
                    return lhs < static_cast<VTLhs>(rhs);
                case LessEqual:
                    return lhs <= static_cast<VTLhs>(rhs);
                case GreaterThan:
                    return lhs > static_cast<VTLhs>(rhs);
                case GreaterEqual:
                    return lhs >= static_cast<VTLhs>(rhs);
                case NotEqual:
                    return lhs != static_cast<VTLhs>(rhs);
            }
        }
        throw std::runtime_error("Unknown compare operation");
    }
    
    /**
     * @brief Generates (or updates) a position list of two join columns, which fulfill the join condition.
     *
     * @tparam VTLhs value type of left hand side column
     * @tparam VTRhs value type of right hand side column
     */
    template<typename VTLhs, typename VTRhs>
    struct CompareColumnPair {
        /**
         * @brief Execute function of this kernel.
         * @param container Convenience structure to store both relations and give easy access to meta data
         * @param positions Pointer reference to resulting position list
         * @param depth Index of the depth-th equation in the theta join
         */
        static void apply(
          /// Container, containing Frames
          Container& container,
          ResultContainer *& positions,
          /// traverse depth
          size_t depth)
        {
            Equation& eq = container.equations.at(depth);
            auto const * lhsData = reinterpret_cast<VTLhs const*>(container.lhs->getColumnRaw(eq.lhsColumnIndex));
            auto const * rhsData = reinterpret_cast<VTRhs const*>(container.rhs->getColumnRaw(eq.rhsColumnIndex));
            
            size_t lhsRowCount = container.lhs->getNumRows();
            size_t rhsRowCount = container.rhs->getNumRows();
            
            if(!positions) {
                positions = new ResultContainer(lhsRowCount * rhsRowCount);
            }
            
            if(depth == 0){
                for(size_t outerLoop = 0; outerLoop < lhsRowCount; ++outerLoop){
                    for(size_t innerLoop = 0; innerLoop < rhsRowCount; ++innerLoop){
                        if(compareValues<VTLhs, VTRhs>(lhsData[outerLoop], rhsData[innerLoop], eq.cmp)){
                            positions->addPosPair(outerLoop, innerLoop);
                        }
                    }
                }
            } else {
                for(uint64_t i = 0; i < positions->size(); ++i){
                    auto [lhsPos, rhsPos] = positions->readNext();
                    if(compareValues<VTLhs, VTRhs>(lhsData[lhsPos], rhsData[rhsPos], eq.cmp)){
                        positions->addPosPair(lhsPos, rhsPos);
                    }
                }
            }
            positions->finalize();
        }
    };
    
//    template<typename VTLhs, typename VTRhs>
//    static void traverseColumnWiseCore(
//      /// Container, containing Frames
//      Container& state,
//      ResultContainer *& positions,
//      /// traverse depth
//      size_t depth)
//    {
//        Equation& eq = state.equations.at(depth);
////        VTLhs const * lhsData = state.lhs->template getColumn<VTLhs>(eq.lhsColumnIndex)->getValues();
////        VTRhs const * rhsData = state.rhs->template getColumn<VTRhs>(eq.rhsColumnIndex)->getValues();
//        VTLhs const * lhsData = reinterpret_cast<VTLhs const*>(state.lhs->getColumnRaw(eq.lhsColumnIndex));
//        VTRhs const * rhsData = reinterpret_cast<VTRhs const*>(state.rhs->getColumnRaw(eq.rhsColumnIndex));
//
//        size_t lhsRowCount = state.lhs->getNumRows();
//        size_t rhsRowCount = state.rhs->getNumRows();
//
//        if(!positions) {
//            positions = new ResultContainer(lhsRowCount * rhsRowCount);
//        }
//
//        if(depth == 0){
//            /// run over whole column
//            for(size_t outerLoop = 0; outerLoop < lhsRowCount; ++outerLoop){
//                for(size_t innerLoop = 0; innerLoop < rhsRowCount; ++innerLoop){
//                    if(compare2<VTLhs, VTRhs>(lhsData[outerLoop], rhsData[innerLoop], eq.cmp)){
////                        std::cout << "true";
//                        positions->addPosPair(outerLoop, innerLoop);
//                    } else {
////                        std::cout << "false";
//                    }
////                    std::cout << std::endl;
//                }
//            }
//        } else {
//            /// only process pre-filtered rows
//            for(uint64_t i = 0; i < positions->size; ++i){
//                auto [lhsPos, rhsPos] = positions->readNext();
//                if(compare2<VTLhs, VTRhs>(lhsData[lhsPos], rhsData[rhsPos], eq.cmp)){
//                    positions->addPosPair(lhsPos, rhsPos);
//                }
//            }
//        }
////        std::cout << "Positions:\n" << *positions->positions << std::endl;
//        positions->finalize();
//    }
//
//    template<typename VTLhs>
//    static void traverseColumnWiseRhs(
//      /// Container, containing Frames
//      Container& state,
//      ResultContainer *& positions,
//      /// traverse depth
//      size_t depth)
//    {
//        /// determine correct value type of the right hand side
//        switch ( state.rhsSchema[state.equations.at(depth).rhsColumnIndex] ){
//            case ValueTypeCode::SI8:
//                traverseColumnWiseCore<VTLhs, int8_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::SI32:
//                traverseColumnWiseCore<VTLhs, int32_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::SI64:
//                traverseColumnWiseCore<VTLhs, int64_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI8:
//                traverseColumnWiseCore<VTLhs, uint8_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI32:
//                traverseColumnWiseCore<VTLhs, uint32_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI64:
//                traverseColumnWiseCore<VTLhs, uint64_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::F32:
//                traverseColumnWiseCore<VTLhs, float>(state, positions, depth);
//                break;
//            case ValueTypeCode::F64:
//                traverseColumnWiseCore<VTLhs, double>(state, positions, depth);
//                break;
//            default:
//                throw std::runtime_error("compareRhs: Unknown value type '" + std::to_string((uint64_t)state.rhsSchema[depth]) + "' found.");
//      }
//    }
//
//    static void traverseColumnWiseLhs(
//      /// Container, containing Frames
//      Container& state,
//      ResultContainer *& positions,
//      /// traverse depth
//      size_t depth)
//    {
//        /// determine correct value type of the left hand side
//        switch ( state.lhsSchema[state.equations.at(depth).lhsColumnIndex] ){
//            case ValueTypeCode::SI8:
//                traverseColumnWiseRhs<int8_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::SI32:
//                traverseColumnWiseRhs<int32_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::SI64:
//                traverseColumnWiseRhs<int64_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI8:
//                traverseColumnWiseRhs<uint8_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI32:
//                traverseColumnWiseRhs<uint32_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::UI64:
//                traverseColumnWiseRhs<uint64_t>(state, positions, depth);
//                break;
//            case ValueTypeCode::F32:
//                traverseColumnWiseRhs<float>(state, positions, depth);
//                break;
//            case ValueTypeCode::F64:
//                traverseColumnWiseRhs<double>(state, positions, depth);
//                break;
//            default:
//                throw std::runtime_error("compareLhs: Unknown value type '" + std::to_string((uint64_t)state.rhsSchema[depth]) + "' found.");
//        }
//    }
//
//    static void traverseColumnWise(
//      /// Container, containing Frames
//      Container& state,
//      ResultContainer *& positions,
//      /// traverse depth
//      size_t depth)
//    {
//        traverseColumnWiseLhs(state, positions, depth);
//    }
    
  public:
    static void apply(Frame*& res, const Frame* lhs, const Frame* rhs, const char** lhsOn, size_t numLhsOn,
                      const char** rhsOn, size_t numRhsOn, CompareOperation* cmp, size_t numCmp) {
        /// @todo get rid of redundant parameters ??
        assert((numLhsOn == numRhsOn and numRhsOn == numCmp) && "incorrect amount of compare values");
        
//        size_t lhsRows = lhs->getNumRows();
//        size_t rhsRows = rhs->getNumRows();
        size_t lhsCols = lhs->getNumCols();
        size_t rhsCols = rhs->getNumCols();
//        auto lhsSchema = lhs->getSchema();
//        auto rhsSchema = rhs->getSchema();
//        auto lhsLabels = lhs->getLabels();
//        auto rhsLabels = rhs->getLabels();
        
        
        /// convenience container holding all relevant data for traversing over both relations
        Container container(lhs, rhs, lhsOn, rhsOn, cmp, numCmp);
        
//        auto resSchema = new ValueTypeCode[lhsCols + rhsCols];
//        auto resLabels = new std::string[lhsCols + rhsCols];
//        for(uint64_t i = 0; i < lhsCols; ++i){
//            resSchema[i] = lhsSchema[i];
//            resLabels[i] = lhsLabels[i];
//        }
//        for(uint64_t i = 0; i < rhsCols; ++i){
//            resSchema[i + lhsCols] = rhsSchema[i];
//            resLabels[i + lhsCols] = rhsLabels[i];
//        }
        
        ResultContainer * resultPositions = nullptr;
    
        
        for(size_t i = 0; i < numCmp; ++i){
//            traverseColumnWise(container, resultPositions, i);
            DeduceValueTypeAndExecute<CompareColumnPair>::apply(
              /// lhs value type
                container.getVTLhs(i),
              /// rhs value type
                container.getVTRhs(i),
              /// parameter of TraverseColumnWise
              container, resultPositions, i
              );
        }
        
        /// write result
        res = DataObjectFactory::create<Frame>(resultPositions->size(), lhsCols + rhsCols,
                                               container.createResultSchema(), container.createResultLabels(), false);
        for(uint64_t i = 0; i < lhsCols; ++i){
            DeduceValueTypeAndExecute<WriteColumn>::apply(container.lhsSchema[i], res, container,
                                                          i, i, true, resultPositions);
        }
        for(uint64_t i = 0; i < rhsCols; ++i){
            DeduceValueTypeAndExecute<WriteColumn>::apply(container.rhsSchema[i], res, container,
                                                          i, i + lhsCols, false, resultPositions);
        }
        
        /// cleanup
        delete resultPositions;
    }
};






#endif //SRC_RUNTIME_LOCAL_KERNELS_THETAJOIN_H
