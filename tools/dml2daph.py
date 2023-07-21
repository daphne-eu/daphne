from antlr4 import *
from DmlLexer import DmlLexer
from DmlParser import DmlParser
from DmlVisitor import DmlVisitor
import argparse
import os

# Context is needed for the following cases:
# 1) Automatic initialization of variables that are only initialized inside an if-else-statement
# 2) Get data type of each variable in order to translate functions correctly from dml to daphneDSL
# 3) Remember functions that are imported or are defined later in the script (otherwise we would try to import them)
# 4) Save return types of function in order to implement dml's stop() function by calling return prematurely
class Context:
    def __init__(self):
        self.if_assigns = set()
        self.else_assigns = set()
        self.already_assigned = set()
        self.in_ifBody = False
        self.in_elseBody = False
        self.data_types = dict()
        self.return_types = {}
        self.current_function = ""
        self.function_names = []
        self.imports = ""
        
class Translator(DmlVisitor):
    def __init__(self):
        self.context = Context()
        
        # Mapping functions from dml to their respective counterparts in daphneDSL
        self.FUNCTIONS_MAP = {
            "matrix": self.matrix_function,
            "ncol": self.ncol_function,
            "nrow": self.nrow_function,
            "t": self.t_function,
            "sum": self.sum_function,
            "ifelse": self.ifelse_function,
            "print": self.print_function,
            "stop": self.stop_function,
            "colMaxs": self.colMaxs_function,
            "rowMaxs": self.rowMaxs_function,
            "colMins": self.colMins_function,
            "rowMins": self.rowMins_function,
            "colSums": self.colSums_function,
            "rowSums": self.rowSums_function,
            "toString": self.toString_function,
            "replace" : self.replace_function,
            "min": self.min_function,
            "max": self.max_function,
            "seq": self.seq_function,
            "diag": self.diag_function,
            "outer": self.outer_function,
            "sqrt": self.sqrt_function,
            "removeEmpty": self.removeEmpty_function,
            "mean": self.mean_function,
            "log": self.log_function,
            "as.integer": self.asInteger_function,
            "as.scalar": self.asScalar_function,
            "as.float": self.asScalar_function,
            "as.matrix": self.asMatrix_function,
            "cbind": self.cbind_function,
            "rbind": self.rbind_function,
            "solve": self.solve_function,
            "abs": self.abs_function,
            "exp": self.exp_function,
            "colIndexMin": self.colIndexMin_function,
            "rowIndexMin": self.rowIndexMin_function,
            "table": self.table_function,
            "time": self.time_function
        }
        
        # TODO Get return types dynamically from code of imported file
        # Return types for builtin dml functions
        self.RETURN_TYPES = {
        	"lm": ["matrix<f64>"],
        	"lmCG": ["matrix<f64>"],
        	"lmDS": ["matrix<f64>"],
        	"dist": ["matrix<f64>"],
        	"components": ["matrix<f64>"]
        }

    def getValueType(self, dtype):
        value_type = dtype.split("<")
        if len(value_type) == 1:
            return ""
        value_type = value_type[1].split(">")[0]
        
        return value_type
    
    # Orders function arguments correctly and extracts values/expressions
    def reorder_args(self, args, correct_order):
        named_args = {k: [v, expr] for k, v, expr in args if k is not None}
        unnamed_args = [[v, expr] for k, v, expr in args if k is None]
        
        ordered_values = []
        ordered_exprs = []
        
        for order in correct_order:
            if order in named_args:
                v, expr = named_args[order]
                ordered_values.append(v)
                ordered_exprs.append(expr)
            elif unnamed_args:
                v, expr = unnamed_args.pop(0)
                ordered_values.append(v)
                ordered_exprs.append(expr)
                
        return ordered_values, ordered_exprs
        
    def matrix_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, "rows", "cols"])
  	
    	dtype = self.inferType(args[0], exprs[0])
    	
    	if dtype in {"si64", "f64", "bool", "str"}:
    		function_call = f"fill(as.f64({args[0]}), {args[1]}, {args[2]})"
    		dtype = "matrix<" + dtype + ">"
    	else:
    		function_call = f"reshape({args[0]}, {args[1]}, {args[2]})"
    		
    	return function_call, dtype
        
    def ncol_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.si64(ncol({args[0]}))"
        dtype = "si64"
        return function_call, dtype

    def nrow_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"as.si64(nrow({args[0]}))"
        dtype = "si64"
        return function_call, dtype

    def t_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"t({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        return function_call, dtype
        
    def sum_function(self, arguments):
        args, exprs = self.reorder_args(arguments, [None])
        function_call = f"sum({args[0]})"
        dtype = self.inferType(args[0], exprs[0])
        dtype = self.getValueType(dtype)
        return function_call, dtype

    def ifelse_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, None, None])
    	function_call = f"{args[0]} ? {args[1]} : {args[2]}"
    	dtype = self.inferType(args[1], exprs[1])
    	return function_call, dtype

    # Handles toString() as well
    def print_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"{args[0]}"
    	
    	# Split combined print statement into seperate statements and handle toString()
    	def split_print(print_str):
    		split_str = print_str.split('+')
    		
    		transformed_strs = []
    		for part in split_str:
    			stripped_part = part.strip()
    			if stripped_part.startswith('toString(') and stripped_part.endswith(')'):
    				var_name = stripped_part[len('toString('):-1]
    				transformed_strs.append(f'print({var_name});')
    			else:
    				transformed_strs.append(f'print({stripped_part});')
    		
    		# Combine all prints
    		all_prints = ' '.join(transformed_strs)
    		return all_prints.rstrip(';')
    	
    	return split_print(function_call), None

    # Prints message and then calls "return" with default values
    def stop_function(self, arguments): 
    	args, exprs = self.reorder_args(arguments, [None])
    	return_string = ", ".join(self.default_value(arg) for arg in self.context.return_types[self.context.current_function])
    	function_call = f"print({args[0]});\nreturn {return_string}"
    	return function_call, None

    def colMaxs_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"aggMax({args[0]}, 1)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def rowMaxs_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"aggMax({args[0]}, 0)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def colMins_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"aggMin({args[0]}, 1)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def rowMins_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"aggMin({args[0]}, 0)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def colSums_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"sum({args[0]}, 1)" 
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def rowSums_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"sum({args[0]}, 0)" 
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype
                
    # Is handled in print()
    def toString_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"toString({args[0]})"
    	dtype = "str"
    	return function_call, dtype

    def replace_function(self, arguments):
        args, exprs = self.reorder_args(arguments, ["target", "pattern", "replacement"])
            
        #args[2] = args[2].strip()
        if args[2].strip() == "0 / 0":
        	args[2] = "nan"
        	
        function_call = f"replace({args[0]}, {args[1]}, {args[2]})"
        dtype = self.inferType(args[0], exprs[0])
        return function_call, dtype
       
    def min_function(self, arguments):
    	function_call = ""
    	dtype = ""
    	if len(arguments) == 1:
    		args, exprs = self.reorder_args(arguments, [None])
    		function_call = f"aggMin({args[0]})"
    		
    		dtype = self.inferType(args[0], exprs[0])
    		dtype = self.getValueType(dtype)
    	elif len(arguments) == 2:
    		args, exprs = self.reorder_args(arguments, [None, None])
    		function_call = f"min({args[0]}, {args[1]})"
    		
    		dtype_arg1 = self.inferType(args[0], exprs[0])
    		dtype_arg2 = self.inferType(args[1], exprs[1])
    		
    		if self.isMatrix(dtype_arg1):
    			dtype = self.getValueType(dtype_arg1)
    		elif self.isMatrix(dtype_arg2):
    			dtype = self.getValueType(dtype_arg2)
    		else:
    			dtype = dtype_arg1
    			
    	return function_call, dtype

    def max_function(self, arguments):
    	function_call = ""
    	dtype = ""
    	if len(arguments) == 1:
    		args, exprs = self.reorder_args(arguments, [None])
    		function_call = f"aggMax({args[0]})"
    		
    		dtype = self.inferType(args[0], exprs[0])
    		dtype = self.getValueType(dtype)
    	elif len(arguments) == 2:
    		args, exprs = self.reorder_args(arguments, [None, None])
    		function_call = f"max({args[0]}, {args[1]})"
    		
    		dtype_arg1 = self.inferType(args[0], exprs[0])
    		dtype_arg2 = self.inferType(args[1], exprs[1])
    		
    		if self.isMatrix(dtype_arg1):
    			dtype = self.getValueType(dtype_arg1)
    		elif self.isMatrix(dtype_arg2):
    			dtype = self.getValueType(dtype_arg2)
    		else:
    			dtype = dtype_arg1
    			
    	return function_call, dtype
        
    # TODO: Handle offset
    def seq_function(self, arguments):
    	function_call = ""
    	dtype = ""
    	if len(arguments) == 2:
    		args, exprs = self.reorder_args(arguments, [None, None])
    		function_call = f"seq({args[0]}, {args[1]}, 1.0)"
    		dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	elif len(arguments) == 3:
    		args, exprs = self.reorder_args(arguments, [None, None, None])
    		function_call = f"seq({args[0]}, {args[1]}, {args[2]})"
    		dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    		
    	return function_call, dtype

    # TODO: implement diagVector
    def diag_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, None])
    	function_call = f"diagMatrix({args[0]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def outer_function(self, arguments):
    	dml_to_daph = {
    		"*": "outerMul",
    		"-": "outerSub",
    		"+": "outerAdd",
    		"/": "outerDiv",
    		"^": "outerPow",
    		"%": "outerMod",
    		"min": "outerMin",
    		"max": "outerMax"
    	}
    	
    	args, exprs = self.reorder_args(arguments, [None, None, None])
    	
    	op = args[2].replace('"', '')
    	
    	function_name = dml_to_daph.get(op)
    	if function_name is None:
    		raise ValueError(f"Unkown operation {op} for outer()!")
    	
    	function_call = f"{function_name}({args[0]}, {args[1]})"
    	dtype = self.inferType(args[0], exprs[0])
    	
    	return function_call, dtype 

    def sqrt_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"sqrt({args[0]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype 

    def removeEmpty_function(self, arguments):
    	function_call = ""
    	args, exprs = self.reorder_args(arguments, ["target", "margin", "select"])
    	
    	args[1] = args[1].replace('"', '')
    	if args[1] == "cols":
    		function_call = f"{args[0]}[[, {args[2]}]]"
    	elif args[1] == "rows":
    		function_call = f"{args[0]}[[{args[2]}, ]]"
    		
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype 

    # TODO: handle case of 2 arguments
    def mean_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"mean({args[0]})"
    	dtype = "f64"
    	return function_call, dtype 

    def log_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"ln({args[0]})"
    	
    	dtype = self.inferType(args[0], exprs[0])
    	# TODO:maby convert si64 to f64
    	
    	return function_call, dtype 

    def asInteger_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"as.si64({args[0]})"
    	dtype = "si64"
    	return function_call, dtype 

    def asScalar_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"as.scalar({args[0]})"
    	
    	dtype = self.inferType(args[0], exprs[0])
    	dtype = self.getValueType(dtype)
    	
    	return function_call, dtype 
        
    def asFloat_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"as.f64({args[0]})"
    	dtype = "f64"
    	return function_call, dtype 

    def asMatrix_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"as.matrix({args[0]})"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def cbind_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, None])
    	function_call = f"cbind({args[0]}, {args[1]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def rbind_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, None])
    	function_call = f"rbind({args[0]}, {args[1]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def solve_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None, None])
    	function_call = f"solve({args[0]}, {args[1]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def abs_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"abs({args[0]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype 

    def exp_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"exp({args[0]})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype 

    def colIndexMin_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"idxMin({args[0]}, 1)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    def rowIndexMin_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None])
    	function_call = f"idxMin({args[0]}, 0)"
    	dtype = "matrix<" + self.inferType(args[0], exprs[0]) + ">"
    	return function_call, dtype

    # TODO: handle multiple named arguments
    def table_function(self, arguments):
    	args, exprs = self.reorder_args(arguments, [None]*len(arguments))
    	args_str = ", ".join(args)
    	function_call = f"ctable({args_str})"
    	dtype = self.inferType(args[0], exprs[0])
    	return function_call, dtype

    def time_function(self, arguments_order, arguments_expr, arguments):
    	function_call = f"now()"
    	dtype = "f64"
    	return function_call, dtype
               
    # Function for indenting the code
    def indent(self, code, level):
    	indentation = "\t" * level
    	lines = code.split("\n")
    	indented_lines = [indentation + line if i > 0 else line for i, line in enumerate(lines)]
    	
    	return "\n".join(indented_lines)
      
    # Checking if a data type is a matrix
    def isMatrix(self, dtype):
    	if "<" in dtype and ">" in dtype:
    		return True
    	else:
    		return False
    	
    # Return values for initialization of variables based on their data type
    def default_value(self, dtype):
    	if dtype == "si64":
    		return "0"
    	elif dtype == "f64":
    		return "0.0"
    	elif dtype == "bool":
    		return "false"
    	elif dtype == "str":
    		return "''"
    	elif self.isMatrix(dtype):
    		value_type = self.getValueType(dtype)
    		if value_type == "si64":
    			return "[0]"
    		elif value_type == "f64":
    			return "[0.0]"
    		elif value_type == "bool":
    			return "[false]"
    		elif value_type == "str":
    			return "['']"
    	else:
    		raise ValueError(f"Unknown data type: {dtype}")
        
    # Determines the data type of a given variable
    def inferType(self, source, ctx):
    	source_type = self.context.data_types.get(source)
    	if source_type is not None:
    		return source_type
    	else:
        	if isinstance(ctx, DmlParser.ConstIntIdExpressionContext):
        		return "si64"
        	elif isinstance(ctx, DmlParser.ConstDoubleIdExpressionContext):
        		return "f64"
        	elif isinstance(ctx, DmlParser.ConstStringIdExpressionContext):
        		return "str"
        	elif isinstance(ctx, DmlParser.ConstTrueExpressionContext) or isinstance(ctx, DmlParser.ConstFalseExpressionContext):
        		return "bool"
        	elif isinstance(ctx, DmlParser.BuiltinFunctionExpressionContext):
        		return self.visitBuiltinFunctionExpression(ctx, type=True)
        	elif isinstance(ctx, DmlParser.AddSubExpressionContext):
        		dtype_left = self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
        		dtype_right = self.inferType(self.visitExpression(ctx.expression(1)), ctx.expression(1))
        		if self.isMatrix(dtype_left):
        			return dtype_left
        		else:
        			return dtype_right
        	elif isinstance(ctx, DmlParser.MultDivExpressionContext):
        		dtype_left = self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
        		dtype_right = self.inferType(self.visitExpression(ctx.expression(1)), ctx.expression(1))
        		if self.isMatrix(dtype_left):
        			return dtype_left
        		else:
        			return dtype_right
        	elif isinstance(ctx, DmlParser.AtomicExpressionContext):
        		return self.inferType(self.visitExpression(ctx.expression()), ctx.expression())
        	elif isinstance(ctx, DmlParser.MatrixMulExpressionContext):
        		return self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
        	elif isinstance(ctx, DmlParser.RelationalExpressionContext):	#TODO check
        		return self.visitRelationalExpression(ctx, type=True)
        	elif isinstance(ctx, DmlParser.UnaryExpressionContext):
        		return self.inferType(self.visitExpression(ctx.expression()), ctx.expression())	# TODO check
        	elif isinstance(ctx, DmlParser.BooleanAndExpressionContext):
        		return "bool"
        	elif isinstance(ctx, DmlParser.PowerExpressionContext):
        		dtype_left = self.inferType(self.visitExpression(ctx.expression(0)), ctx.expression(0))
        		dtype_right = self.inferType(self.visitExpression(ctx.expression(1)), ctx.expression(1))
        		if self.isMatrix(dtype_left):
        			return dtype_left
        		else:
        			return dtype_right
        	elif isinstance(ctx, DmlParser.BooleanNotExpressionContext):
        		return "bool"
        	elif isinstance(ctx, DmlParser.ModIntDivExpressionContext):
        		return "si64"
        	elif isinstance(ctx, DmlParser.BooleanOrExpressionContext):
        		return "bool"
        	elif "[" in source and "]" in source:
        		# TODO: Allow direct initialization of matrices (i.e. matrix = [1,2,3,4])
                # TODO: Allow slice operation
                # TODO: Allow nested function calls as indexes
        		
        		# Get the name of the variable (i.e. 'ext' from 'ext[i,0]')
        		name, idx = source.split("[")
        		dtype = self.context.data_types.get(name)
        		if ":" in idx:
        			return dtype
        		else:
        			dtype = self.getValueType(dtype)
        			return dtype
        	elif source == "nan" or source == "inf":
        		return "f64"

        	raise ValueError(f"Unkown data type of {source}")  
      
    def retrieve_function_parameters(self, ctx):
    	target = None
    	if ctx:
    		try:
    			target = self.visitDataIdentifier(ctx.dataIdentifier())
    		except:
    			pass
    			
    	function_name = ctx.ID().getText()
    	args = []
    	for param_expr in ctx.parameterizedExpression():
    		param_name, param_val = self.visitParameterizedExpression(param_expr)
    		args.append([param_name, param_val, param_expr.expression()])
    		
    	return target, function_name, args

    # Handle implicit import of builtin dml scripts 
    def handleImplicitImport(self, script_name):
    	# Get the file path and namespace 
    	
    	file_name = script_name + ".dml"
    	file_path = "../thirdparty/systemds/scripts/builtin/" + file_name
    	
    	try:
    		with open(file_path, "r") as f:
    			dml_code = f.read()
    			
    		# Run the translator recursively on the imported file
    		lexer = DmlLexer(InputStream(dml_code))
    		tokens = CommonTokenStream(lexer)
    		parser = DmlParser(tokens)
    		tree = parser.programroot()
    		
    		# Traverse parse tree and translate it to daphneDSL
    		translator = Translator()
    		daph_code = translator.visitProgramroot(tree)
    		
    		# Write the translated code to a new file and change from .dml to .daph
    		file_path = os.path.basename(file_path)
    		file_path = os.path.splitext(file_path)[0]
    		file_path = file_path + ".daph"
    		
    		with open(file_path, "w") as f:
    			f.write(daph_code)

    	except FileNotFoundError:
    		raise ValueError(f"Unknown function: {script_name}")
    		
    	# Translate the import statement
    	translated_import = f'import "{file_path}";\n';
    
        # TODO: Retrieve return data type of function automatically from file-string
    	return translated_import, "matrix<f64>"		

    def create_function_call(self, function_name, arguments):
    	function_call = ""
    	dtype = ""
    	
    	arg_list = [arg[1] for arg in arguments]
    	arg_str = ", ".join(arg_list)
	
	# Cases: builtin function, function inside this file, dml script that was not explicitly imported (explicit imports are handled in visitImportStatement)
    	if function_name in self.FUNCTIONS_MAP:
    		function_call, dtype = self.FUNCTIONS_MAP[function_name](arguments)
    	elif function_name in self.context.function_names:
    		function_call = f"{function_name}({arg_str})"
    		dtype = self.context.return_types[function_name]
    	else: 
    		import_str, dtype = self.handleImplicitImport(function_name)
    		self.context.imports += import_str
    		function_call = f"{function_name}.m_{function_name}({arg_str})"	# Assumes that function name is "m_" + script name
    		
    	return function_call, dtype

    # TODO: handle multiple targets (and retrieve their respective data types)
    def handle_assignment(self, target, function_call, dtype):
        assignment = f"{function_call};"
        if target is not None:
            if not self.context.in_ifBody and not self.context.in_elseBody:
                self.context.already_assigned.add(target)
            elif self.context.in_ifBody:
                self.context.if_assigns.add(target)
            elif self.context.in_elseBody:
                self.context.else_assigns.add(target)
            self.context.data_types[target] = dtype
            assignment = f"{target} = {function_call};"
        return assignment

    def visitFunctionCallAssignmentStatement(self, ctx: DmlParser.FunctionCallAssignmentStatementContext):
        target, function_name, arguments = self.retrieve_function_parameters(ctx)
        function_call, dtype = self.create_function_call(function_name, arguments)
        assignment = self.handle_assignment(target, function_call, dtype)

        return assignment

    # Translate function call with multiple return values
    def visitFunctionCallMultiAssignmentStatement(self, ctx: DmlParser.FunctionCallMultiAssignmentStatementContext):
        # Retrieve the targets of the function
        targets = [self.visitDataIdentifier(di) for di in ctx.dataIdentifier() if di is not None]

        # Retrieve function parameters
        _, function_name, arguments = self.retrieve_function_parameters(ctx)

        # Create the function call
        function_call, dtype = self.create_function_call(function_name, arguments)

        # Handle assignments (to add targets to list of initialized variables)
        i = 0
        for target in targets:
            self.handle_assignment(target, function_call, dtype)
            self.context.data_types[target] = dtype[i]
            i += 1

        # Format multiple assignments
        targets_str = ', '.join(targets)
        multi_assignment = f"{targets_str} = {function_call};"
            
        return multi_assignment

    # Translate builtin-function call
    def visitBuiltinFunctionExpression(self, ctx: DmlParser.BuiltinFunctionExpressionContext, type=False):
        target, function_name, arguments = self.retrieve_function_parameters(ctx)
        function_call, dtype = self.create_function_call(function_name, arguments)

        if type:
        	return dtype
        return function_call
        
    # TODO Only takes Imports and Function-definitions as entrypoints: allow other types as well
    # Starting point of the translation
    def visitProgramroot(self, ctx: DmlParser.ProgramrootContext):
    	# Get the names of 1) all the functions inside the file and 2) of all imported functions
    	for child in ctx.children:
    		if isinstance(child, DmlParser.InternalFunctionDefExpressionContext):
    			function_name = child.ID().getText()
    			self.context.function_names.append(function_name)
    			self.context.return_types[function_name] = []
    			
    			output_params = []
    			for param in child.typedArgNoAssign():
    				self.context.return_types[function_name].append(self.visitTypedArgNoAssign(param, 0))
    		elif isinstance(child, DmlParser.ImportStatementContext):
    			function_name = ctx.filePath.text[1:-1]
    			self.context.function_names.append(function_name)
    			self.context.return_types[function_name] = self.context.RETURN_TYPES[function_name]	# TODO: get return data types dynamically
    			
    	translated_code = ""
    	for child in ctx.children:
    		if isinstance(child, DmlParser.InternalFunctionDefExpressionContext):
    			self.context.current_function = child.ID().getText()
    			translated_code += self.visitInternalFunctionDefExpression(child)
    		elif isinstance(child, DmlParser.ImportStatementContext):
    			translated_code += self.visitImportStatement(child)
    			
    	return self.context.imports + translated_code

    # Identify exact Statement-Type and call responsible function
    def visitStatement(self, ctx: DmlParser.StatementContext):
        if isinstance(ctx, DmlParser.AssignmentStatementContext):
        	return self.visitAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.FunctionCallAssignmentStatementContext):
        	return self.visitFunctionCallAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.ForStatementContext):
        	return self.visitForStatement(ctx)
        elif isinstance(ctx, DmlParser.IterablePredicateContext):
        	return self.visitIterablePredicate(ctx)
        elif isinstance(ctx, DmlParser.IfStatementContext):
        	return self.visitIfStatement(ctx)
        elif isinstance(ctx, DmlParser.FunctionCallMultiAssignmentStatementContext):
        	return self.visitFunctionCallMultiAssignmentStatement(ctx)
        elif isinstance(ctx, DmlParser.WhileStatementContext):
        	return self.visitWhileStatement(ctx)
        elif isinstance(ctx, DmlParser.ParForStatementContext):
        	return self.visitParForStatement(ctx)
        elif isinstance(ctx, DmlParser.AccumulatorAssignmentStatementContext):
        	return self.visitAccumulatorAssignmentStatement(ctx)
        else:
        	raise ValueError(f"Unkown statement {ctx}")
        	return "UNKOWN_STATEMENT"
        
    # Identify exact Expression-Type and call responsible function
    def visitExpression(self, ctx: DmlParser.ExpressionContext):
        if isinstance(ctx, DmlParser.AddSubExpressionContext):
        	return self.visitAddSubExpression(ctx)
        elif isinstance(ctx, DmlParser.DataIdExpressionContext):
        	return self.visitDataIdExpression(ctx)
        elif isinstance(ctx, DmlParser.BuiltinFunctionExpressionContext):
        	return self.visitBuiltinFunctionExpression(ctx)
        elif isinstance(ctx, DmlParser.MultDivExpressionContext):
        	return self.visitMultDivExpression(ctx)
        elif isinstance(ctx, DmlParser.AtomicExpressionContext):
        	return self.visitAtomicExpression(ctx)
        elif isinstance(ctx, DmlParser.MatrixMulExpressionContext):
        	return self.visitMatrixMulExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstIntIdExpressionContext):
        	return self.visitConstIntIdExpression(ctx)
        elif isinstance(ctx, DmlParser.RelationalExpressionContext):
        	return self.visitRelationalExpression(ctx)
        elif isinstance(ctx, DmlParser.UnaryExpressionContext):
        	return self.visitUnaryExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanAndExpressionContext):
        	return self.visitBooleanAndExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstDoubleIdExpressionContext):
        	return self.visitConstDoubleIdExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstFalseExpressionContext):
        	return self.visitConstFalseExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstTrueExpressionContext):
        	return self.visitConstTrueExpression(ctx)
        elif isinstance(ctx, DmlParser.ConstStringIdExpressionContext):
        	return self.visitConstStringIdExpression(ctx)
        elif isinstance(ctx, DmlParser.PowerExpressionContext):
        	return self.visitPowerExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanNotExpressionContext):
        	return self.visitBooleanNotExpression(ctx)
        elif isinstance(ctx, DmlParser.ModIntDivExpressionContext):
        	return self.visitModIntDivExpression(ctx)
        elif isinstance(ctx, DmlParser.BooleanOrExpressionContext):
        	return self.visitBooleanOrExpression(ctx)
        elif isinstance(ctx, DmlParser.ParameterizedExpressionContext):
        	return self.visitParameterizedExpression(ctx)
        else:
        	raise ValueError(f"Unkown expression {ctx}")
        	return "UNKOWN_EXPRESSION"

    # Defining a function
    def visitInternalFunctionDefExpression(self, ctx: DmlParser.InternalFunctionDefExpressionContext):
        function_name = ctx.ID().getText()

        # Get parameters
        input_params = []
        for param in ctx.typedArgAssign():
            input_params.append(self.visitTypedArgAssign(param))
        input_params_string = ', '.join(input_params)

        # Get data / value types of return values
        output_params = []
        for param in ctx.typedArgNoAssign():
            output_params.append(self.visitTypedArgNoAssign(param, 0))
        output_params_string = ', '.join(output_params)

        # Get variable names of return values
        output_names = []
        for param in ctx.typedArgNoAssign():
            output_names.append(self.visitTypedArgNoAssign(param, 1))
        output_names_string = ', '.join(output_names)
        
        # Get function body
        function_body = []
        for statement in ctx.statement():
            function_body.append(self.visitStatement(statement))
        function_body_string = '\n'.join(function_body)  
        indented_function_body = self.indent(function_body_string, 1)

        # Construct the function
        translated_function = f"def {function_name}({input_params_string}) -> {output_params_string}" + " {\n"
        translated_function += f"\t{indented_function_body}" 
        translated_function += "\n\treturn " + f"{output_names_string};" + "\n}\n\n"
        
        return translated_function

        
    # Retrieve input parameters
    def visitTypedArgAssign(self, ctx: DmlParser.TypedArgAssignContext):
        param_type = self.visitMl_type(ctx.ml_type())
        param_name = ctx.ID().getText()
        
        # Return the parameter (optionally with initial value)
        if ctx.paramVal is not None:
            param_val = self.visitExpression(ctx.paramVal)
            if param_val == "NaN" or param_val == "Inf":
            	param_val = param_val.lower()
            translated_param = f"{param_name}:{param_type} /*= {param_val}*/"
        else:
            translated_param = f"{param_name}:{param_type}"
        
          
        # Specify data type of variable in global dict
        self.context.data_types[param_name] = param_type
            
        return translated_param

    # Retrieve output parameters
    def visitTypedArgNoAssign(self, ctx: DmlParser.TypedArgNoAssignContext, types):
    	param_type = self.visitMl_type(ctx.ml_type())
    	param_name = ctx.ID().getText()
    	
    	# Return either type or name (depending on specified parameter "types")
    	if types == 0:
    		translated_param = f"{param_type}"
    	else:
    		translated_param = f"{param_name}"
    		
    	return translated_param
    	
    # Retrieve DataType and ValueType
    def visitMl_type(self, ctx:DmlParser.Ml_typeContext):
    	if ctx.dataType() is not None:					# i.e. "Matrix[Double] X"
    		data_type = self.visitDataType(ctx.dataType())
    		value_type = self.visitDataType(ctx.valueType())
    		translated_type = f"{data_type}<{value_type}>"
    	else:								# i.e. "Double X"
    		value_type = self.visitDataType(ctx.valueType())
    		translated_type = f"{value_type}"
    		
    	return translated_type
    
    # Retrieve DataType
    def visitDataType(self, ctx: DmlParser.DataTypeContext):
        dtype = ctx.getText().lower()
        if dtype == "integer":
        	dtype = "si64"
        elif dtype == "string":
        	dtype = "str"
        elif dtype == "boolean":
        	dtype = "bool"
        elif dtype == "double":
        	dtype = "f64"
        return dtype
    
    # Get name of target via visitDataIdentifier
    def visitDataIdExpression(self, ctx: DmlParser.DataIdExpressionContext):
    	data_identifier = self.visitDataIdentifier(ctx.dataIdentifier())
    	return data_identifier
    
    # TODO Allow slice operation
    # TODO Allow nested function calls as indexes
    # Get name of target
    def visitDataIdentifier(self, ctx: DmlParser.DataIdentifierContext):
    	identifier = ctx.getText()
    	
    	# Check if identifier contains indexing
    	if '[' in identifier and ']' in identifier:
    		# Split the string into three parts: matrix name, indices, rest
    		matrix_name, indices = identifier.split('[')
    		indices, rest = indices.split(']')
    		
    		# Split the indices into row and column and decrement the indices
    		idx = indices.split(',')
    		if len(idx) == 1:
    			index = idx[0]
    			try:
	    			index = str(int(index.strip()) - 1)
	    		except ValueError:
	    			if index != "" and index != " ":
	    				index = index + " - 1"
	    				
	    		# Construct the identifier
	    		identifier = f'{matrix_name}[{index}]{rest}'
    		else:
    			row_index, col_index = idx
    			
    			# Decrement the indices (i.e. "matrix[a,1]" mapped to "matrix[a-1,0]")
	    		try:
	    			row_index = str(int(row_index.strip()) - 1)
	    		except ValueError:
	    			if row_index != "" and row_index != " ":
	    				row_index = row_index + " - 1"
	    		try:
	    			col_index = str(int(col_index.strip()) - 1)
	    		except ValueError:
	    			if col_index != "" and col_index != " ":
	    				col_index = col_index + " - 1"
    		
	    		# Construct the identifier
	    		identifier = f'{matrix_name}[{row_index}, {col_index}]{rest}'
    	
    	return identifier     
    
    # Simply visit the sub-expression and return its translation in parenthesis
    def visitAtomicExpression(self, ctx: DmlParser.AtomicExpressionContext):
    	return "(" + self.visitExpression(ctx.expression()) + ")"
    	
    # i.e. "x + y" or "x - y"
    def visitAddSubExpression(self, ctx: DmlParser.AddSubExpressionContext):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	# Get the operator
    	op = ctx.op.text
    	
    	# Construct the expression
    	translated_expression = f"{left} {op} {right}"
    	
    	return translated_expression
    
    # i.e. "X @ Y"
    def visitMatrixMulExpression(self, ctx: DmlParser.MatrixMulExpressionContext):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	return f"{left} @ {right}"
    	
    # i.e. "x * y" or "x / y"
    def visitMultDivExpression(self, ctx: DmlParser.MultDivExpressionContext):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	# Get the operator
    	op = ctx.op.text
    	
    	# Construct the expression
    	translated_expression = f"{left} {op} {right}"
    	
    	return translated_expression
    	
    # i.e. "x > y"
    def visitRelationalExpression(self, ctx: DmlParser.RelationalExpressionContext, type=False):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	left_dtype = self.inferType(left, ctx.expression(0))
    	right_dtype = self.inferType(right, ctx.expression(1))

    	dtype = "bool"
    	if self.isMatrix(left_dtype) and not self.isMatrix(right_dtype):
    		dtype = left_dtype
    	elif not self.isMatrix(left_dtype) and self.isMatrix(right_dtype):
    		dtype = right_dtype
    	elif self.isMatrix(left_dtype) and self.isMatrix(right_dtype):
    		dtype = left_dtype
    	
    	# Get the operator 
    	op = ctx.op.text
    	
    	# Construct the expression
    	translated_expression = f"{left} {op} {right}"
    	
    	if type == True:
    		return dtype
    	
    	return translated_expression
    	
    # i.e. translate "-x" to "0.0-x"
    def visitUnaryExpression(self, ctx: DmlParser.UnaryExpressionContext):
    	# Visit the operand
    	operand = self.visitExpression(ctx.expression())
    	
    	# Get the operator
    	op = ctx.op.text
    	
    	# Construct the unary expression
    	try:
    		num = float(operand)
    		if op == "-":
    			num = 0.0-num
    		translated_expression = f"({num})"
    	except ValueError:
    		translated_expression = f"(0.0{op}{operand})"
    	
    	return translated_expression
    
    # i.e. "x && y"
    def visitBooleanAndExpression(self, ctx: DmlParser.BooleanAndExpressionContext):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	# Get the operator
    	op = ctx.op.text
    	
    	# Construct the expression (Daphne: "&&", Dml: "&")
    	translated_expression = f"{left} {op}{op} {right}"
    	
    	return translated_expression
    	
    # i.e. "x || y"
    def visitBooleanOrExpression(self, ctx: DmlParser.BooleanOrExpressionContext):
        # Visit the left and right expressions
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))

        # Get the operator
        op = ctx.op.text

        # Construct translated expression
        translated_expression = f"{left} {op}{op} {right}"
        
        return translated_expression
    
    # i.e. "x ^ y"
    def visitPowerExpression(self, ctx: DmlParser.PowerExpressionContext):
    	# Visit the left and right expressions
    	left = self.visitExpression(ctx.expression(0))
    	right = self.visitExpression(ctx.expression(1))
    	
    	# Construct the expression
    	translated_expression = f"{left} ^ {right}"
    	
    	return translated_expression
    	
    # Dml supports "!x" but Daphne does not (therefore translating "!x" to "x == false"
    def visitBooleanNotExpression(self, ctx: DmlParser.BooleanNotExpressionContext):
        # Visit the expression
        expr = self.visitExpression(ctx.expression())

        # Construct the translated expression
        translated_expression = f"({expr} == false)"
        
        return translated_expression
           
    # Modulus and Integer division
    def visitModIntDivExpression(self, ctx: DmlParser.ModIntDivExpressionContext):
        # Visit the left and right expressions
        left = self.visitExpression(ctx.expression(0))
        right = self.visitExpression(ctx.expression(1))

        # Get the operator
        op = ctx.op.text

        # Translate mod and integer division
        if op == "%/%":
        	translated_expression = f"round({left} / {right})"
        else:
        	translated_expression = f"{left} % {right}"
        
        return translated_expression
        
    # Float constant
    def visitConstDoubleIdExpression(self, ctx: DmlParser.ConstDoubleIdExpressionContext):
    	# Get double constant
    	const = ctx.DOUBLE().getText()
    	
    	# Change from scientific notation to decimal values
    	num = float(const)
    	num = "{:f}".format(num)
    	
    	return f"{num}" 
    	
    # Integer constant
    def visitConstIntIdExpression(self, ctx: DmlParser.ConstIntIdExpressionContext):
    	# Get the integer constant
    	const = ctx.INT().getText()
    	
    	return f"{const}"
    
    # Boolean constant
    def visitConstFalseExpression(self, ctx: DmlParser.ConstFalseExpressionContext):
    	return 'false'
    
    # Boolean constant
    def visitConstTrueExpression(self, ctx: DmlParser.ConstTrueExpressionContext):
    	return 'true'
    	
    # String constant
    def visitConstStringIdExpression(self, ctx: DmlParser.ConstStringIdExpressionContext):
    	# Get the string constant
    	dml_string = ctx.STRING().getText()
    	
    	return dml_string
    	
    # Input parameters for a function
    def visitParameterizedExpression(self, ctx: DmlParser.ParameterizedExpressionContext, init_vals = True):
        param_name = ctx.paramName.text if ctx.paramName is not None else None
        param_val = self.visitExpression(ctx.expression())
        
        if param_val == "NaN" or param_val == "Inf":
        	param_val = param_val.lower()
        
        # Check if the parameter has a name
        if param_name is not None and init_vals == True:
        	return param_name, param_val
        else:
        	return None, param_val	
        		 
    # Translate simple assignment-statement
    def visitAssignmentStatement(self, ctx: DmlParser.AssignmentStatementContext):
    	target = self.visitDataIdentifier(ctx.dataIdentifier())
    	source = self.visitExpression(ctx.expression())
    	
    	if source == "NaN" or source == "Inf":
    		source = source.lower()
    		
    	# Add target variable to global lists (needed for initialization of variables before if-else)
    	if not self.context.in_ifBody and not self.context.in_elseBody:
    		self.context.already_assigned.add(target)
    	elif self.context.in_ifBody:
    		self.context.if_assigns.add(target)
    	elif self.context.in_elseBody:
    		self.context.else_assigns.add(target)
    		
    	# Specify data type of target in global dict
    	if "[" in target and "]" in target:
    		target_name = target.split("[")[0]
    		dtype = self.context.data_types.get(target_name)
    		if dtype is not None:
    			# TODO Add const booleans
    			if isinstance(ctx.expression(), DmlParser.ConstIntIdExpressionContext):
    				source += ".0"
    				return f"{target} = as.matrix({source});"
    			elif isinstance(ctx.expression(), DmlParser.ConstDoubleIdExpressionContext):
    				return f"{target} = as.matrix({source});"
    		
    	self.context.data_types[target] = self.inferType(source, ctx.expression())
    	
    	#print(f"{target} {source}")
    	#print(f"{target}: {self.context.data_types[target]}"

    	return f"{target} = {source};"
    
    # i.e. "x += y" (is not supported in daphne, therefore needs to be translated to "x = x + y")
    def visitAccumulatorAssignmentStatement(self, ctx: DmlParser.AccumulatorAssignmentStatementContext):
        # Get the target
        target = self.visitDataIdentifier(ctx.dataIdentifier())

        # Get the operator
        op = ctx.op.text

        # Get the source
        source = self.visitExpression(ctx.expression())

        # Create the assignment statement (turn a += b into a = a + b)
        assignment = f"{target} {op[1]} {target} {op[0]} {source};"

        return assignment
    
    # For-Loop
    def visitForStatement(self, ctx: DmlParser.ForStatementContext):
    	# Get the iteration variable
    	iter_var = ctx.ID().getText()
    	
    	# Get the iterable predicate
    	iterable_pred = self.visitIterablePredicate(ctx.iterablePredicate())
    	
    	# Get the parameters
    	params = [param for param in (self.visitStrictParameterizedExpression(expr) for expr in ctx.strictParameterizedExpression()) if param is not None]
    	params_string = ', '.join(params)
    	
    	# Get the body of the for-loop
    	body = [self.visitStatement(stmt) for stmt in ctx.statement()]
    	body_string = '\n'.join(body)
    	
    	# Construct the translated for-loop
    	translated_for_loop = f"for({iter_var} in {iterable_pred}{params_string})" + " {\n"
    	translated_for_loop += self.indent(f"\t{body_string}", 1)
    	translated_for_loop += "\n}"
    	
    	return translated_for_loop
    
    # Parfor-Loop (does not exist in daphne, therefore translated to normal for-loop)
    def visitParForStatement(self, ctx: DmlParser.ParForStatementContext):
    	# Get the iteration variable
    	iter_var = ctx.ID().getText()
    	
    	# Get the iterable predicate
    	iterable_pred = self.visitIterablePredicate(ctx.iterablePredicate())
    	
    	# Get the parameters
    	params = [param for param in (self.visitStrictParameterizedExpression(expr) for expr in ctx.strictParameterizedExpression()) if param is not None]
    	params_string = ', '.join(params)
    	
    	# Get the body of the for-loop
    	body = [self.visitStatement(stmt) for stmt in ctx.statement()]
    	body_string = '\n'.join(body)
    	
    	# Construct the translated for-loop
    	translated_for_loop = f"for({iter_var} in {iterable_pred}{params_string})" + " {\n"
    	translated_for_loop += self.indent(f"\t{body_string}", 1)
    	translated_for_loop += "\n}"
    	
    	return translated_for_loop
    
    # If-Else
    def visitIfStatement(self, ctx: DmlParser.IfStatementContext):
    	# Get the predicate expression
    	predicate = self.visitExpression(ctx.expression())
    	
    	# Get the if body
    	self.context.in_ifBody = True
    	if_body_statements = [self.visitStatement(stmt) for stmt in ctx.ifBody]
    	self.context.in_ifBody = False
    	
    	# Get the else body
    	self.context.in_elseBody = True
    	else_body_statements = [self.visitStatement(stmt) for stmt in ctx.elseBody]
    	self.context.in_elseBody = False
    	
    	# Initialize variables that are only initialized inside the if-else-block
    	common_assigns = self.context.if_assigns.intersection(self.context.else_assigns)
    	common_assigns -= self.context.already_assigned
    	common_assigns_str = "\n".join([f"{var} = {self.default_value(self.context.data_types[var])};" for var in common_assigns])
    	
    	# Turn if and else body into string
    	if_body = '\n'.join(if_body_statements)
    	else_body = '\n'.join(else_body_statements)
    	
    	# Clear the variable list (otherwise the same variables would be initialized again before the next if-statement)
    	self.context.if_assigns.clear()
    	self.context.else_assigns.clear()
    	
    	# Construct the if statement
    	if_statement = f"if ({predicate})" + " {\n"
    	if_statement += self.indent(f"\t{if_body}", 1)
    	if_statement += "\n}"
    	
    	# If there is an else body, add it to the if statement
    	if else_body_statements:
    		if_statement += " else {\n"
    		if_statement += self.indent(f"\t{else_body}", 1)
    		if_statement += "\n}"
    		
    	return common_assigns_str + "\n" + if_statement + "\n"
    		
    # While-Loop
    def visitWhileStatement(self, ctx: DmlParser.WhileStatementContext):
        # Get the expression
        expression = self.visitExpression(ctx.expression())

        # Get the body of the while loop
        body_statements = [self.visitStatement(stmt) for stmt in ctx.statement()]
        body = '\n'.join(body_statements)

        # Construct the translated while loop
        translated_while_loop = f"while ({expression})" + " {\n"
        translated_while_loop += self.indent(f"\t{body}", 1)
        translated_while_loop += "\n}"
        
        return translated_while_loop
        
    # Translate iterable predicate
    def visitIterablePredicate(self, ctx: DmlParser.IterablePredicateContext):
    	if isinstance(ctx, DmlParser.IterablePredicateColonExpressionContext):
    		from_exp = self.visitExpression(ctx.expression(0))
    		to_exp = self.visitExpression(ctx.expression(1))
    		
    		iterable_pred = f"{from_exp}:{to_exp}"
    		
    		return iterable_pred
    	elif isinstance(ctx, DmlParser.IterablePredicateSeqExpressionContext):
    		from_exp = self.visitExpression(ctx.expression(0))
    		to_exp = self.visitExpression(ctx.expression(1))
    		
    		iterable_pred = f"{from_exp}:{to_exp}"
    		if ctx.increment is not None:
    			increment = self.visitExpression(ctx.increment)
    			iterable_pred += f":{increment}"
    		
    		return iterable_pred
    	else:
    		raise ValueError(f"Unkown expression {ctx} for iterable predicate")
    		return "UNKOWN_PREDICATE"
    		
    # TODO: Change function name to alias (if namespace is specified)
    # Translating import statement
    def visitImportStatement(self, ctx: DmlParser.ImportStatementContext):
    	# Get the file path and namespace
    	file_path = "../systemds/" + ctx.filePath.text[1:-1]	# Assuming daphne and systemds are in the same directory + translator is in /daphne
    	namespace = ctx.namespace.text if ctx.namespace is not None else None
    	
    	try:
    		with open(file_path, "r") as f:
    			dml_code = f.read()
    			
    		# Run the translator recursively on the imported file
    		lexer = DmlLexer(InputStream(dml_code))
    		tokens = CommonTokenStream(lexer)
    		parser = DmlParser(tokens)
    		tree = parser.programroot()
    		
    		# Traverse parse tree and translate it to daphneDSL
    		translator = Translator()
    		daph_code = translator.visitProgramroot(tree)
    		
    		# Write the translated code to a new file and change from .dml to .daph
    		file_path = os.path.basename(file_path)
    		file_path = os.path.splitext(file_path)[0]
    		file_path = file_path + ".daph"
    		with open(file_path, "w") as f:
    			f.write(daph_code)
    	except FileNotFoundError:
    		print(f"File not found! {file_path}")
    		
    	# Translate the import statement
    	translated_import = f"import {file_path}"
    	if namespace is not None:
    		translated_import += f' as "{namespace}"'
    	translated_import += ";\n"
    	
    	return translated_import

# Translator from Dml to DaphneDSL
def translate(dml_code):
    # Generate parse tree
    lexer = DmlLexer(InputStream(dml_code))
    tokens = CommonTokenStream(lexer)
    parser = DmlParser(tokens)
    tree = parser.programroot()

    # Traverse parse tree and translate it to daphneDSL
    translator = Translator()
    daph_code = translator.visitProgramroot(tree)

    return daph_code

def get_daphne_filename(dml_filename):
    # Replace the .dml extension with .daph and store it in the directory of the translator script
    base_name = os.path.basename(dml_filename)
    base_name = base_name.replace('.dml', '.daph')
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'translated_files')
    return os.path.join(output_dir, base_name)

# Reading .dml script and writing .daph script
if __name__ == "__main__":
    # Get dml file path via command line
    parser = argparse.ArgumentParser(description="Translate Dml file to DaphneDSL.")
    parser.add_argument('dml_filename', type=str, help='Path to the Dml file to translate')
    args = parser.parse_args()

    with open(args.dml_filename, 'r') as f:
        dml_code = f.read()

    daph_code = translate(dml_code)

    daph_filename = get_daphne_filename(args.dml_filename)
    with open(daph_filename, 'w') as f:
        f.write(daph_code)
