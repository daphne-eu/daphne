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

 #include <runtime/local/kernels/EwBinarySca.h>

 #include <tags.h>
 
 #include <catch.hpp>
 
 #include <cstdint>
 
 #define TEST_NAME(opName) "EwBinarySca (" opName ")"
 #define VALUE_TYPES double, uint32_t
 
 template <BinaryOpCode opCode, typename VT> void checkEwBinarySca(VT lhs, VT rhs, VT exp) {
     CHECK(EwBinarySca<opCode, VT, VT, VT>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<VT, VT, VT>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 template <BinaryOpCode opCode> void checkEwBinarySca(std::string lhs, std::string rhs, int64_t exp) {
     CHECK(EwBinarySca<opCode, int64_t, std::string, std::string>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<int64_t, std::string, std::string>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 template <BinaryOpCode opCode> void checkEwBinarySca(FixedStr16 lhs, FixedStr16 rhs, int64_t exp) {
     CHECK(EwBinarySca<opCode, int64_t, FixedStr16, FixedStr16>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<int64_t, FixedStr16, FixedStr16>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 template <BinaryOpCode opCode> void checkEwBinarySca(UnorderedDictionaryEncodedString lhs, UnorderedDictionaryEncodedString rhs, int64_t exp) {
     CHECK(EwBinarySca<opCode, int64_t, UnorderedDictionaryEncodedString, UnorderedDictionaryEncodedString>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<int64_t, UnorderedDictionaryEncodedString, UnorderedDictionaryEncodedString>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 template <BinaryOpCode opCode> void checkEwBinarySca(OrderedDictionaryEncodedString lhs, OrderedDictionaryEncodedString rhs, int64_t exp) {
     CHECK(EwBinarySca<opCode, int64_t, OrderedDictionaryEncodedString, OrderedDictionaryEncodedString>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<int64_t, OrderedDictionaryEncodedString, OrderedDictionaryEncodedString>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 template <typename VT> void checkEwBinarySca(VT lhs, VT rhs, std::string exp) {
     CHECK(EwBinarySca<BinaryOpCode::CONCAT, std::string, VT, VT>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<std::string, VT, VT>(BinaryOpCode::CONCAT, lhs, rhs, nullptr) == exp);
 }
 
 
 template <BinaryOpCode opCode, typename VT, typename VTRes> void checkEwBinarySca(VT lhs, VT rhs, VTRes exp) {
     CHECK(EwBinarySca<opCode, VTRes, VT, VT>::apply(lhs, rhs, nullptr) == exp);
     CHECK(ewBinarySca<VTRes, VT, VT>(opCode, lhs, rhs, nullptr) == exp);
 }
 
 
 
 
 // ****************************************************************************
 // Arithmetic
 // ****************************************************************************
 
 TEMPLATE_TEST_CASE(TEST_NAME("add"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(0, 0, 0);
     checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(0, 1, 1);
     checkEwBinarySca<BinaryOpCode::ADD, VT, VT>(1, 2, 3);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("mul"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(0, 0, 0);
     checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(0, 1, 0);
     checkEwBinarySca<BinaryOpCode::MUL, VT, VT>(2, 3, 6);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("div"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::DIV, VT, VT>(0, 3, 0);
     checkEwBinarySca<BinaryOpCode::DIV, VT, VT>(6, 3, 2);
 }
 
 // ****************************************************************************
 // Comparisons
 // ****************************************************************************
 
 TEMPLATE_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(0, 0, 1);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(3, 3, 1);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VT>(3, 5, 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("eq"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = int64_t;
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT(" "), VT(""), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Valentia"), VT("Valencia"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Parisiorum Lute"), VT("Paris"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Roma"), VT("Roma"), 1);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Palma"), VT("PalmadeMallorca"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Pompeii"), VT("Pompei"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Thessalonica"), VT("Thessaloniki"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Regium Lepidi"), VT("Reggio Emilia"), 0);
     checkEwBinarySca<BinaryOpCode::EQ, VT, VTRes>(VT("Augusta Trevero"), VT("Augusta Trevero"), 1);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(0, 0, 0);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(3, 3, 0);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VT>(3, 5, 1);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("neq"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = int64_t;
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT(" "), VT(""), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Valentia"), VT("Valencia"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Parisiorum Lute"), VT("Paris"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Roma"), VT("Roma"), 0);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Palma"), VT("PalmadeMallorca"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Pompeii"), VT("Pompei"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Thessalonica"), VT("Thessaloniki"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Regium Lepidi"), VT("Reggio Emilia"), 1);
     checkEwBinarySca<BinaryOpCode::NEQ, VT, VTRes>(VT("Augusta Trevero"), VT("Augusta Trevero"), 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::LT, VT, VT>(1, 1, 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VT>(1, 3, 1);
     checkEwBinarySca<BinaryOpCode::LT, VT, VT>(4, 2, 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("lt"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = int64_t;
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Tenochtitlan"), VT("Coixtlahuaca"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Tlacozauhtitlan"), VT("Quiauhteopan"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Texcoco"), VT("Temazcaltepec"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Cuauhchinanco"), VT("Tepecoacuilco"), 1);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Tlacopan"), VT("Tlacozauhtitlan"), 1);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Tochtepec"), VT("Tochpan"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Quetzaltecolotl"), VT("Quetzaltecolotl"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Chimalpopoca"), VT("Chimalpopoc"), 0);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT("Huitzilopochtl"), VT("Huitzilopochtli"), 1);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT(""), VT("abc"), 1);
     checkEwBinarySca<BinaryOpCode::LT, VT, VTRes>(VT(""), VT(""), 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("le"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::LE, VT, VT>(1, 1, 1);
     checkEwBinarySca<BinaryOpCode::LE, VT, VT>(1, 3, 1);
     checkEwBinarySca<BinaryOpCode::LE, VT, VT>(4, 2, 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::GT, VT, VT>(1, 1, 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VT>(1, 3, 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VT>(4, 2, 1);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("gt"), TAG_KERNELS, ALL_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = int64_t;
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Tenochtitlan"), VT("Coixtlahuaca"), 1);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Tlacozauhtitlan"), VT("Quiauhteopan"), 1);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Texcoco"), VT("Temazcaltepec"), 1);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Cuauhchinanco"), VT("Tepecoacuilco"), 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Tlacopan"), VT("Tlacozauhtitlan"), 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Tochtepec"), VT("Tochpan"), 1);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Quetzaltecolotl"), VT("Quetzaltecolotl"), 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Chimalpopoca"), VT("Chimalpopoc"), 1);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT("Huitzilopochtl"), VT("Huitzilopochtli"), 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT(""), VT("abc"), 0);
     checkEwBinarySca<BinaryOpCode::GT, VT, VTRes>(VT(""), VT(""), 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("ge"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::GE, VT, VT>(1, 1, 1);
     checkEwBinarySca<BinaryOpCode::GE, VT, VT>(1, 3, 0);
     checkEwBinarySca<BinaryOpCode::GE, VT, VT>(4, 2, 1);
 }
 
 // ****************************************************************************
 // Min/max
 // ****************************************************************************
 
 TEMPLATE_TEST_CASE(TEST_NAME("min"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(2, 2, 2);
     checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(2, 3, 2);
     checkEwBinarySca<BinaryOpCode::MIN, VT, VT>(3, 0, 0);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("max"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(2, 2, 2);
     checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(2, 3, 3);
     checkEwBinarySca<BinaryOpCode::MAX, VT, VT>(3, 0, 3);
 }
 
 // ****************************************************************************
 // Logical
 // ****************************************************************************
 
 TEMPLATE_TEST_CASE(TEST_NAME("and"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(0, 0, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(0, 1, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(1, 0, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(1, 1, 1);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(0, 2, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(2, 0, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(2, 2, 1);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(0, -2, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(-2, 0, 0);
     checkEwBinarySca<BinaryOpCode::AND, VT, VT>(-2, -2, 1);
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("or"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(0, 0, 0);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(0, 1, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(1, 0, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(1, 1, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(0, 2, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(2, 0, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(2, 2, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(0, -2, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(-2, 0, 1);
     checkEwBinarySca<BinaryOpCode::OR, VT, VT>(-2, -2, 1);
 }
 
 // ****************************************************************************
 // String ops
 // ****************************************************************************
 TEMPLATE_TEST_CASE(TEST_NAME("concat"), TAG_KERNELS, FIXED_SIZE_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = std::string;
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Pachakutiy"), VT(" Inka Yupanki"),
                                                       VTRes("Pachakutiy Inka Yupanki"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Tawantinsuyu:"), VT(" Chinchay Suyu"),
                                                       VTRes("Tawantinsuyu: Chinchay Suyu"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Túpac Inca"), VT(" Yupanqui"), VTRes("Túpac Inca Yupanqui"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT(""), VT("abc"), VTRes("abc"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT(""), VT(""), VTRes(""));
 }
 
 TEMPLATE_TEST_CASE(TEST_NAME("concat"), TAG_KERNELS, FLEXIBLE_SIZE_STRING_VALUE_TYPES) {
     using VT = TestType;
     using VTRes = VT;
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Pachakutiy"), VT(" Inka Yupanki"),
                                                       VT("Pachakutiy Inka Yupanki"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Tawantinsuyu:"), VT(" Chinchay Suyu"),
                                                       VT("Tawantinsuyu: Chinchay Suyu"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT("Túpac Inca"), VT(" Yupanqui"), VT("Túpac Inca Yupanqui"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT(""), VT("abc"), VT("abc"));
     checkEwBinarySca<BinaryOpCode::CONCAT, VT, VTRes>(VT(""), VT(""), VT(""));
 }
 
 // ****************************************************************************
 // Invalid op-code
 // ****************************************************************************
 
 TEMPLATE_TEST_CASE(TEST_NAME("some invalid op-code"), TAG_KERNELS, VALUE_TYPES) {
     using VT = TestType;
     CHECK_THROWS(ewBinarySca<VT, VT, VT>(static_cast<BinaryOpCode>(999), 0, 0, nullptr));
 }