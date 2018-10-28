import sys, time, argparse
import aux_lib as aux


def main(argv):
    terms_path = './color_terms.csv'
    output_folder = './result/'
    verbose = False
    img_path = './images.csv'
    multiprocessing = True

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input csv file or directory with images")
    parser.add_argument("terms", type=str, help="Path to terms csv file")
    parser.add_argument("output", type=str, help="path to output folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output on STOUT")
    parser.add_argument("-j", "--json", action="store_true", help="produce jsons for web view")# choices=['only', 'additional'])
    args = parser.parse_args()

    if args.input[-3:] == 'csv':
        inputmode = 'csv'
    else:
        inputmode = 'dir'

    aux.start(args.input, inputmode, args.terms, args.output, args.verbose, args.json)

    print_time((time.time() - start_time))
    print('# # # # # # # # # # # # # # # # # # ')


def print_time(t):
    # hours
    hours = t // 3600
    # remaining seconds
    t -= hours * 3600
    # minutes
    minutes = t // 60
    # remaining seconds
    seconds = t - (minutes * 60)
    # total time
    print('%s hours %s minutes %.2f seconds' % (int(hours), int(minutes), seconds))


if __name__ == "__main__":
    main(sys.argv[1:])
