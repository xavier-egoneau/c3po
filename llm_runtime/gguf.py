"""
Lecture des métadonnées d'un fichier GGUF directement dans l'en-tête, sans charger
le modèle (ni llama.cpp). Sert à connaître le nombre de couches et le contexte
d'entraînement avant de calculer les paramètres d'inférence.

On ne matérialise que les quelques clés utiles ; les gros tableaux (vocabulaire
tokenizer, ~150k entrées) sont sautés en avançant le pointeur de fichier.
"""

from __future__ import annotations
import struct
from pathlib import Path

_MAGIC = b"GGUF"

# Types de valeurs GGUF (cf. spec)
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)

_SCALAR_FMT = {
    _U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
    _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d",
}
_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR_FMT.items()}

# Clés qu'on cherche (suffixe pour être indépendant du préfixe d'architecture)
_WANTED_SUFFIXES = (".block_count", ".context_length", ".embedding_length")


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) < n:
            raise ValueError("fichier GGUF tronqué")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")

    def scalar(self, vtype: int):
        return struct.unpack(_SCALAR_FMT[vtype], self.read(_SCALAR_SIZE[vtype]))[0]

    def skip_value(self, vtype: int) -> None:
        """Avance le pointeur au-delà d'une valeur sans la matérialiser."""
        if vtype in _SCALAR_SIZE:
            self.f.seek(_SCALAR_SIZE[vtype], 1)
        elif vtype == _STR:
            self.f.seek(self.u64(), 1)
        elif vtype == _ARR:
            itype = self.u32()
            count = self.u64()
            if itype in _SCALAR_SIZE:
                self.f.seek(_SCALAR_SIZE[itype] * count, 1)
            elif itype == _STR:
                for _ in range(count):
                    self.f.seek(self.u64(), 1)
            else:
                raise ValueError(f"type de tableau GGUF inconnu : {itype}")
        else:
            raise ValueError(f"type GGUF inconnu : {vtype}")


def read_metadata(path: str | Path) -> dict:
    """
    Retourne les métadonnées scalaires utiles d'un GGUF
    (general.architecture + les clés `*.block_count` / `*.context_length` /
    `*.embedding_length`). Lève ValueError si le fichier n'est pas un GGUF valide.
    """
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.read(4) != _MAGIC:
            raise ValueError("ce n'est pas un fichier GGUF")
        version = r.u32()
        if version < 2:
            raise ValueError(f"version GGUF non supportée : {version}")
        _tensor_count = r.u64()
        kv_count = r.u64()

        meta: dict = {}
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            if key == "general.architecture" and vtype == _STR:
                meta[key] = r.string()
            elif key.endswith(_WANTED_SUFFIXES) and vtype in _SCALAR_FMT:
                meta[key] = r.scalar(vtype)
            else:
                r.skip_value(vtype)
        return meta


def model_shape(path: str | Path) -> dict:
    """
    Dimensions exploitables d'un modèle GGUF, ou des None si illisibles.
    Clés : arch, n_layers, n_ctx_train, n_embd.
    """
    try:
        meta = read_metadata(path)
    except (ValueError, OSError):
        return {"arch": None, "n_layers": None, "n_ctx_train": None, "n_embd": None}

    arch = meta.get("general.architecture")

    def by_suffix(suffix: str):
        for k, v in meta.items():
            if k.endswith(suffix):
                return v
        return None

    return {
        "arch": arch,
        "n_layers": by_suffix(".block_count"),
        "n_ctx_train": by_suffix(".context_length"),
        "n_embd": by_suffix(".embedding_length"),
    }
