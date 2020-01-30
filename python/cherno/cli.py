#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Dec 1, 2017
# @Filename: cli.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego

import argparse
import os
import sys

from cherno.main import math


def main():

    # An example of how to write a command line parser that works with the
    # main.math function. For more details on how to use argparse, start with
    # this tutorial: http://bit.ly/2SGDf7h

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description='Performs an arithmetic operation.')

    parser.add_argument('VALUE1', type=float, help='The first operand')
    parser.add_argument('OPERATOR', type=str, help='The operator [+, -, *, /]')
    parser.add_argument('VALUE2', type=float, help='The second operand')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='sets verbose mode')

    args = parser.parse_args()

    result = math(args.VALUE1, args.VALUE2, arith_operator=args.OPERATOR)

    if args.verbose:
        print('{} {} {} = {}'.format(args.VALUE1, args.OPERATOR, args.VALUE2, result))
    else:
        print(result)


if __name__ == '__main__':

    main()
