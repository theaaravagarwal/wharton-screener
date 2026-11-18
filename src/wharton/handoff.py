"""Private password-encrypted credential handoff; never part of a public data export."""

import argparse
import base64
import getpass
import json
import os
import secrets
import string
import tempfile
from pathlib import Path

from dotenv import dotenv_values

LIMIT = 65536


def _private_read(path):
    path = Path(path)
    if any(parent.is_symlink() for parent in path.absolute().parents):
        raise ValueError("Private destination must not traverse symlink directories")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("Credential source must be a private regular file (chmod 600)")
    if path.stat().st_size > LIMIT:
        raise ValueError("Credential file exceeds handoff size limit")
    return path.read_bytes()


def _write_new(path, content):
    """Publish a complete private file atomically; hard-link creation refuses overwrite."""
    path = Path(path)
    if any(parent.is_symlink() for parent in path.absolute().parents):
        raise ValueError("Private destination must not traverse symlink directories")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            os.chmod(temporary, 0o600)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _cipher(password, salt):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    # Fixed parameters prevent a malicious envelope from requesting unbounded KDF resources.
    key = Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(password.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(key))


def build_private_handoff(source, encrypted_path, password_path):
    """Encrypt only EODHD_API_KEY; save a random 8-character password separately."""
    from io import StringIO

    encrypted_path, password_path = Path(encrypted_path), Path(password_path)
    bundle_root = encrypted_path.parent.resolve()
    if password_path.resolve().is_relative_to(bundle_root):
        raise ValueError("Password file must be outside the encrypted bundle directory")
    if any(p.exists() or p.is_symlink() for p in (encrypted_path, password_path)):
        raise ValueError("Handoff destination already exists; refusing overwrite")
    text = _private_read(source).decode("utf-8")
    key = dotenv_values(stream=StringIO(text), interpolate=False).get("EODHD_API_KEY")
    if not key or any(c in key for c in "\r\n\x00"):
        raise ValueError("Source needs a one-line EODHD_API_KEY")
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    plaintext = f"EODHD_API_KEY='{escaped}'\n".encode("utf-8")
    groups = (string.ascii_uppercase, string.ascii_lowercase, string.digits, "!@#$%^&*()-_=+?")
    alphabet = "".join(groups)
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(8))
        if all(any(character in group for character in password) for group in groups):
            break
    salt = os.urandom(16)
    envelope = json.dumps({"version": 1, "salt": base64.b64encode(salt).decode("ascii"),
                           "token": _cipher(password, salt).encrypt(plaintext).decode("ascii")})
    # A failure publishing ciphertext leaves only a private password file, never plaintext.
    _write_new(password_path, (password + "\n").encode("ascii"))
    _write_new(encrypted_path, envelope.encode("ascii"))
    return {"encrypted_file": str(encrypted_path), "separate_password_file": str(password_path)}


def unlock(encrypted_path, destination, password):
    """Authenticate before publishing plaintext; wrong passwords leave no destination."""
    from io import StringIO

    from cryptography.fernet import InvalidToken

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Local credential destination exists; refusing overwrite")
    path = Path(encrypted_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("Invalid encrypted credential file")
    try:
        envelope = json.loads(path.read_bytes())
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"version", "salt", "token"}
            or type(envelope["version"]) is not int
            or envelope["version"] != 1
        ):
            raise ValueError
        salt = base64.b64decode(envelope["salt"], validate=True)
        if len(salt) != 16 or not isinstance(envelope["token"], str):
            raise ValueError
        plaintext = _cipher(password, salt).decrypt(envelope["token"].encode("ascii"))
        values = dotenv_values(stream=StringIO(plaintext.decode("utf-8")), interpolate=False)
        key = values.get("EODHD_API_KEY")
        if set(values) != {"EODHD_API_KEY"} or not key or any(c in key for c in "\r\n\x00"):
            raise ValueError
    except (InvalidToken, ValueError, TypeError, KeyError, UnicodeError):
        raise ValueError("Credential authentication failed; no credentials written") from None
    _write_new(destination, plaintext)
    return "Credentials unlocked locally (permissions 0600)."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encrypted_file", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if not os.isatty(0):
        parser.error("Unlock requires an interactive terminal; do not pass passwords in arguments")
    try:
        print(unlock(args.encrypted_file, args.destination, getpass.getpass("Handoff password: ")))
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
