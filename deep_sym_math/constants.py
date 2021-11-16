SYM_BASE_URL = 'https://dl.fbaipublicfiles.com/SymbolicMathematics/data/'

SYM_URLS = {
    'prim_fwd': SYM_BASE_URL + 'prim_fwd.tar.gz',
    'prim_bwd': SYM_BASE_URL + 'prim_bwd.tar.gz',
    'prim_ibp': SYM_BASE_URL + 'prim_ibp.tar.gz',
    'ode1': SYM_BASE_URL + 'ode1.tar.gz',
    'ode2': SYM_BASE_URL + 'ode2.tar.gz',
}

SPECIAL_TOKENS = ['<s>', '</s>', '<pad>', '(', ')']
SPECIAL_WORDS = SPECIAL_TOKENS + [
    f'<SPECIAL_{i}>' for i in range(len(SPECIAL_TOKENS), 10)
]

OPERATORS = {
    # Elementary functions
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'pow': 2,
    'rac': 2,
    'inv': 1,
    'pow2': 1,
    'pow3': 1,
    'pow4': 1,
    'pow5': 1,
    'sqrt': 1,
    'exp': 1,
    'ln': 1,
    'abs': 1,
    'sign': 1,
    # Trigonometric Functions
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'cot': 1,
    'sec': 1,
    'csc': 1,
    # Trigonometric Inverses
    'asin': 1,
    'acos': 1,
    'atan': 1,
    'acot': 1,
    'asec': 1,
    'acsc': 1,
    # Hyperbolic Functions
    'sinh': 1,
    'cosh': 1,
    'tanh': 1,
    'coth': 1,
    'sech': 1,
    'csch': 1,
    # Hyperbolic Inverses
    'asinh': 1,
    'acosh': 1,
    'atanh': 1,
    'acoth': 1,
    'asech': 1,
    'acsch': 1,
    # Derivative
    'derivative': 2,
    # custom functions
    'f': 1,
    'g': 2,
    'h': 3,
}
