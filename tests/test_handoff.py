import base64
import json
import stat

import pytest
from dotenv import dotenv_values

from wharton.handoff import build_private_handoff, unlock


@pytest.fixture
def envelope(tmp_path):
    source = tmp_path / "source.env"
    source.write_text("EODHD_API_KEY='synthetic-only-key'\nOTHER_SECRET=do-not-copy\n")
    source.chmod(0o600)
    encrypted = tmp_path / "bundle" / "key.enc"
    password_file = tmp_path / "password.txt"
    result = build_private_handoff(source, encrypted, password_file)
    return source, encrypted, password_file, result


def test_round_trip_only_key_and_private_permissions(envelope, tmp_path):
    _, encrypted, password_file, result = envelope
    password = password_file.read_text().strip()
    assert len(password) == 8
    assert any(c.isupper() for c in password)
    assert any(c.islower() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(not c.isalnum() for c in password)
    assert password not in encrypted.read_text()
    assert "synthetic-only-key" not in encrypted.read_text()
    assert "synthetic-only-key" not in str(result)
    destination = tmp_path / "recipient" / ".env"
    unlock(encrypted, destination, password)
    assert dict(dotenv_values(destination)) == {"EODHD_API_KEY": "synthetic-only-key"}
    for path in (encrypted, password_file, destination):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_wrong_password_tamper_and_overwrite_fail(envelope, tmp_path):
    _, encrypted, password_file, _ = envelope
    destination = tmp_path / ".env"
    with pytest.raises(ValueError, match="authentication"):
        unlock(encrypted, destination, "wrong-password")
    assert not destination.exists()
    body = json.loads(encrypted.read_text())
    body["token"] = body["token"][:-8] + "tampered"
    encrypted.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="authentication"):
        unlock(encrypted, destination, password_file.read_text().strip())
    assert not destination.exists()
    destination.write_text("preserve")
    with pytest.raises(ValueError, match="refusing overwrite"):
        unlock(encrypted, destination, "anything")
    assert destination.read_text() == "preserve"


def test_password_cannot_be_inside_bundle(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        build_private_handoff(tmp_path / "missing", tmp_path / "bundle/key.enc",
                              tmp_path / "bundle/password.txt")


@pytest.mark.parametrize("payload", [[], {"version": True, "salt": "x", "token": "x"},
                                     {"version": 1, "salt": "x", "token": "x", "n": 2**50}])
def test_malformed_envelope_rejected_without_kdf(tmp_path, payload, monkeypatch):
    encrypted = tmp_path / "bad.enc"
    encrypted.write_text(json.dumps(payload))
    monkeypatch.setattr("wharton.handoff._cipher", lambda *args: pytest.fail("KDF ran"))
    with pytest.raises(ValueError, match="authentication"):
        unlock(encrypted, tmp_path / ".env", "password")


def test_source_and_destination_symlinks_rejected(envelope, tmp_path):
    source, encrypted, password_file, _ = envelope
    link = tmp_path / "source-link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="private"):
        build_private_handoff(link, tmp_path / "next/key.enc", tmp_path / "next-password")
    target = tmp_path / "target"
    target.mkdir()
    link.unlink()
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        unlock(encrypted, link / ".env", password_file.read_text().strip())
    assert not (target / ".env").exists()


def test_authenticated_payload_cannot_import_other_settings(envelope, tmp_path):
    from wharton.handoff import _cipher

    _, encrypted, password_file, _ = envelope
    password = password_file.read_text().strip()
    body = json.loads(encrypted.read_text())
    salt = base64.b64decode(body["salt"])
    body["token"] = _cipher(password, salt).encrypt(
        b"EODHD_API_KEY=synthetic\nOTHER_SETTING=unexpected\n"
    ).decode("ascii")
    encrypted.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="authentication"):
        unlock(encrypted, tmp_path / ".env", password)
    assert not (tmp_path / ".env").exists()
