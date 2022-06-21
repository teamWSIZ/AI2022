SPECIAL = '-.,;…:()!!„”?'


def create_language_dict(filename: str):
    d = dict()
    for c in SPECIAL:
        d[c] = len(d)
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            for w in words:
                print(w)
                while len(w) > 0 and w[-1] in SPECIAL:
                    w = w[:-1]
                while len(w) > 0 and w[0] in SPECIAL:
                    w = w[1:]
                if w not in d.keys():
                    d[w] = len(d)
    return d


def convert_to_tokens(tekst: str, d):
    result = []
    for w in tekst.split():
        while len(w) > 0 and w[0] in SPECIAL:
            result.append(d[w[0]])

        if w not in d.keys(): continue
        result.append(d[w])
    return result


if __name__ == '__main__':
    dct = create_language_dict('lalka-tom-pierwszy.txt')
    for k in dct.keys():
        print(k)
    print(convert_to_tokens('natrętne kwiaciarki otyłej matki', dct))
