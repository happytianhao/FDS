"""Strict JSON Schema construction helpers."""


def obj(**properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def text(size=500):
    return {"type": "string", "maxLength": size}


def enum(*values):
    return {"type": "string", "enum": list(values)}


def array(items, limit=16):
    return {"type": "array", "items": items, "maxItems": limit}
