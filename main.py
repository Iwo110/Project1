import argparse

from multitool import greet, calc, __version__

def main():
    parser = argparse.ArgumentParser(description="MultiTool Skeleton")
    subparsers = parser.add_subparsers(dest="command", required=True)

    greet_parser = subparsers.add_parser("greet", help="Greet the user")
    greet_parser.add_argument("name", help="Name of the user")

    calc_parser = subparsers.add_parser("calc", help="Perform a simple addition")
    calc_parser.add_argument("a", type=int, help="First number")
    calc_parser.add_argument("b", type=int, help="Second number")

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if args.command == "greet":
        greet.run(args)
    elif args.command == "calc":
        calc.run(args)

if __name__ == "__main__":
    main()
