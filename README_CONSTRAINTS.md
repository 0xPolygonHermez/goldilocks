## COMPILE:

make constr_eval

## EXECUTE:

./constr_eval <AST.json> <logN>

## EXAMPLE :

./constr_eval ./constr/ast_1.json 10

This ejecutin will evaluate the constraints of the ast_1.json in a trace of 1024 rows and will output the hash of the resultant matrix.

## IMPORTANT about AST inputs:

Still need to check the AST files generated, please use by now:
* test_ast.json (small example)
* q_polynomial.json (large real case)


## NOTE: This implementation is not focus in performance!

