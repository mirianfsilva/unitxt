def separate_inside_and_outside_square_brackets(s):
    """Separates the content inside and outside the first level of square brackets in a string.

    Allows text before the first bracket and nested brackets within the first level. Raises a ValueError for:
    - Text following the closing bracket of the first bracket pair
    - Unmatched brackets
    - Multiple bracket pairs at the same level

    :param s: The input string to be parsed.
    :return: A tuple (outside, inside) where 'outside' is the content outside the first level of square brackets,
             and 'inside' is the content inside the first level of square brackets. If there are no brackets,
             'inside' will be None.
    """
    start = s.find("[")
    end = s.rfind("]")

    # Handle no brackets
    if start == -1 and end == -1:
        return s, None

    # Validate brackets
    if start == -1 or end == -1 or start > end:
        raise ValueError("Illegal structure: unmatched square brackets.")

    outside = s[:start]
    inside = s[start + 1 : end]
    after = s[end + 1 :]

    # Check for text after the closing bracket
    if len(after.strip()) != 0:
        raise ValueError(
            "Illegal structure: text follows after the closing square bracket."
        )

    return outside, inside


# Formal definition of query:
#  query -> assignment (, assignment)*
#  assignment -> name_value = term
#  term -> name_value | name_value[query] | [ term (, term)* ]


def consume_name_val(instring: str) -> tuple:
    name_val = ""
    for char in instring:
        if char in "[],=":
            break
        name_val += char
    instring = instring[len(name_val) :].strip()
    name_val = name_val.strip()
    sign = 1
    if name_val.startswith("-"):
        sign = -1
        name_val = name_val[1:]
    if name_val.isdigit():
        return (sign * int(name_val), instring)
    if name_val.replace(".", "", 1).isdigit() and name_val.count(".") < 2:
        return (sign * float(name_val), instring)
    if sign == -1:
        name_val = "-" + name_val
    return (name_val, instring)


def consume_term(instring: str, return_dict: bool) -> tuple:
    orig_instring = instring
    if instring.startswith("["):
        toret = []
        instring = instring[1:].strip()
        (term, instring) = consume_term(instring, return_dict)
        toret.append(term)
        while instring.startswith(","):
            (term, instring) = consume_term(instring[1:].strip(), return_dict)
            toret.append(term)
        if not instring.startswith("]"):
            raise ValueError(f"malformed list in: {orig_instring}")
        instring = instring[1:].strip()
        if not return_dict:
            toret = orig_instring[: len(orig_instring) - len(instring)]
        return (toret, instring)

    (name_val, instring) = consume_name_val(instring)
    if instring.startswith("["):
        (quey, instring) = consume_query(instring[1:].strip(), False)
        if not instring.startswith("]"):
            raise ValueError(f"malformed query in: {orig_instring}")
        instring = instring[1:].strip()
        toret = orig_instring[: len(orig_instring) - len(instring)]
        return (toret, instring)
    return (name_val, instring)


def consume_assignment(instring: str, return_dict: bool) -> tuple:
    orig_instring = instring
    (name_val, instring) = consume_name_val(instring)
    if (
        name_val is None
        or isinstance(name_val, int)
        or isinstance(name_val, float)
        or len(name_val) == 0
    ):
        raise ValueError(f"malformed key in assignment that starts: {orig_instring}")
    if not instring.startswith("="):
        raise ValueError(f"malformed assignment in: {orig_instring}")
    (term, instring) = consume_term(instring[1:].strip(), return_dict)
    if (term is None) or not (
        isinstance(term, int) or isinstance(term, float) or len(term) > 0
    ):
        raise ValueError(f"malformed assignment in: {orig_instring}")
    if return_dict:
        return ({name_val: term}, instring)
    toret = orig_instring[: len(orig_instring) - len(instring)]
    return (toret, instring)


def consume_query(instring: str, return_dict: bool) -> tuple:
    (toret, instring) = consume_assignment(instring.strip(), return_dict)
    while instring.startswith(","):
        instring = instring[1:].strip()
        (assignment, instring) = consume_assignment(instring.strip(), return_dict)
        if return_dict:
            toret = {**toret, **assignment}
        else:
            toret = toret + "," + assignment
    return (toret, instring)


def parse_key_equals_value_string_to_dict(query: str) -> dict:
    instring = query
    qu, _ = consume_query(instring, True)
    return qu

    # """Parses a query string of the form 'key1=value1,key2=value2,...' into a dictionary.

    # The function converts numeric values into integers or floats as appropriate, and raises an
    # exception if the query string is malformed or does not conform to the expected format.

    # :param query: The query string to be parsed.
    # :return: A dictionary with keys and values extracted from the query string, with spaces stripped from keys.
    # """
    # result = {}
    # kvs = split_within_depth(query, dellimiter=",")
    # if len(kvs) == 0:
    #     raise ValueError(
    #         f'Illegal query: "{query}" should contain at least one assignment of the form: key1=value1,key2=value2'
    #     )
    # for kv in kvs:
    #     kv = kv.strip()
    #     key_val = split_within_depth(kv, dellimiter="=")
    #     if (
    #         len(key_val) != 2
    #         or len(key_val[0].strip()) == 0
    #         or len(key_val[1].strip()) == 0
    #     ):
    #         raise ValueError(
    #             f'Illegal query: "{query}" with wrong assignment "{kv}" should be of the form: key=value.'
    #         )
    #     key, val = key_val[0].strip(), key_val[1].strip()
    #     if val.isdigit():
    #         result[key] = int(val)
    #     elif val.replace(".", "", 1).isdigit() and val.count(".") < 2:
    #         result[key] = float(val)
    #     else:
    #         try:
    #             result[key] = parse_list_string(val)
    #         except:
    #             result[key] = val

    # return result


def parse_list_string(s: str):
    """Parses a query string of the form 'val1,val2,...' into a list."""
    instring = s
    term, instring = consume_term(instring, True)
    if len(instring.strip()) > 0:
        raise ValueError(f"Illegal list structure in {s}")
    return term

    # start = s.find("[")
    # end = s.rfind("]")

    # Handle no brackets
    # if start == -1 and end == -1:
    #     return s

    # Validate brackets
    # if start == -1 or end == -1 or start > end:
    #     raise ValueError("Illegal structure: unmatched square brackets.")

    # before = s[:start].strip()
    # inside = s[start + 1 : end].strip()
    # after = s[end + 1 :].strip()

    # # Check for text after the closing bracket
    # if len(before) != 0 or len(after) != 0:
    #     raise ValueError(
    #         "Illegal structure: text follows before or after the closing square bracket."
    #     )
    # splitted = split_within_depth(inside.strip(), dellimiter=",", forbbiden_chars=["="])
    # return [s.strip() for s in splitted]
