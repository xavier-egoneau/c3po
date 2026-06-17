import sys


def main(argv):
    if len(argv) != 2:
        print("usage: python cli.py <fichier>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        print(len(handle.read().split()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
