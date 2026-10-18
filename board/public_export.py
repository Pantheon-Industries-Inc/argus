"""Copy JSON for public transport after private source/freshness checks.

Access attributes and recognizable credential values are withheld. This is not a
publication-rights decision. Other prose, native fields and public citations stay intact.
Keep original inputs and operational build receipts in private storage.
"""
from __future__ import annotations

import hashlib
import base64
import ipaddress
import json
import re
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import parse_qsl, unquote, urlsplit


PRIVATE_KEYS = frozenset({
    '_run', '_replaced_label', '_supplements', '_verification', '_withheld_checks', '_carried_verdicts',
    '_uploader_field_path', '_uploader_location_values', '_reader_private_error', '_reader_public_what', '_reader_public_error',
    '_billing_receipt', '_authority_history_index',
})
# These are access mechanisms, not source identities or source-proof digests.
_SECRETS = frozenset({
    'apikey', 'accesstoken', 'refreshtoken', 'idtoken', 'token', 'password', 'passwd',
    'secret', 'secretkey', 'clientsecret', 'secretaccesskey', 'accesskeyid', 'sessiontoken',
    'authorization', 'proxyauthorization', 'cookie', 'setcookie', 'credentials',
    'credential', 'privatekey', 'signedurl', 'presignedurl', 'accessurl',
    'requestheaders', 'authheaders', 'hftoken', 'huggingfacetoken', 'authtoken', 'apitoken',
    'awssessiontoken', 'githubtoken', 'bearertoken', 'openrouterkey', 'openaikey', 'anthropickey',
    'jwt', 'csrf', 'csrftoken', 'sentrydsn', 'connectionstring', 'otpseed', 'pass', 'pwd',
    'privatekeypem', 'secretkeybase', 'apikeys', 'secrets', 'tokens',
})
# Source locations resolve private preparation trees before this projection runs.
_PRIVATE_LOCATIONS = frozenset({
    'sourcelocation', 'rawpath', 'localpath', 'retainedroot', 'retainedevidenceroot',
    'episodedir', 'exampledir', 'jobdir', 'cachedir', 'sourcerowmapfile',
})
_LOCATIONS = frozenset({
    'source', 'sources', 'path', 'paths', 'file', 'files', 'filename', 'directory', 'dir',
    'root', 'packed', 'src', 'location', 'sourceroot', 'sourcepath', 'sourcefile', 'sourcedirectory',
    'archivepath', 'archive', 'url', 'urls', 'uri', 'uris', 'href', 'link', 'links',
    'publicbase', 'publiccitation', 'citation',
})
_URL_QUERY_SECRETS = _SECRETS | {'key', 'signature', 'sig', 'session', 'sessionid', 'code',
                              'keypairid', 'policy', 'googleaccessid'}
_ACCESS_URI_PREFIXES = ('file:', 's3:', 'gs:', 'az:', 'abfs:', 'ssh:', 'sftp:', 'ftp:')
_DROP = object()

# These reader-owned note maps use source files as their keys. Keys inside the
# recorded fields remain native identifiers, including absolute HDF paths.
_NOTE_SOURCE_MAPS = frozenset({
    'recordedmetadata', 'recordedepisodemetadata', 'recordedannotationtables', 'outsidefiles',
    'unreadableannotationtablesources',
})
_NOTE_SOURCE_VALUES = frozenset({'recordedannotationdatasource'})
_NOTE_CLAIMS = frozenset({'unassigneddatasetmetadataclaims'})
_READER_CONTEXT_MAPS = _NOTE_SOURCE_MAPS | {'uploadernotes', 'recordedsensorfields'}
_ATTRIBUTE_LOCATIONS = frozenset({'sourcedirectory', 'sourcelocation'})
_METADATA_LIMIT_SUFFIX = (' and its recorded metadata exceed the prompt note limit; complete original rows '
                          'and source claims remain in the episode context and uploader notes.')
_ANNOTATION_CODES_SUFFIX = (' has unresolved recorded annotation codes; '
                            'all original rows and table claims are retained as uploader notes.')


def _name(key: str) -> str:
    return re.sub('[^a-z0-9]', '', key.lower())


def _credential_key(key: str, *, native=False, owner='') -> bool:
    """Access attributes use semantic suffixes, not a growing provider-name list."""
    name = _name(key)
    if not native and name == 'key' and _name(owner) == 'kaggle':
        return True  # Kaggle's service credential object, not an arbitrary native key.
    if native and name in ('token', 'tokens'):
        return False
    if name in _SECRETS or name in ('auth', 'bearer', 'passphrase'):
        return True
    if name.endswith(('password', 'passwd', 'passphrase', 'secret', 'secretkey',
                      'privatekey', 'apikey', 'apitoken', 'accesstoken', 'refreshtoken',
                      'authtoken', 'bearertoken', 'sessiontoken', 'bottoken', 'accesskey')):
        return True
    # A session ID, signature, tokenizer symbol or field key is not proof of an
    # access mechanism. Credential values are checked independently below.
    return False


_CREDENTIAL_VALUE = re.compile(
    r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----|'
    r'(?<![A-Za-z0-9])(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|'
    r'github_pat_[A-Za-z0-9_]{40,}|sk-ant-[A-Za-z0-9_-]{20,}|'
    r'xox[baprs]-[A-Za-z0-9-]{20,})(?![A-Za-z0-9])|'
    r'\b(?:Proxy-)?Authorization\s*:\s*[A-Za-z][A-Za-z0-9_-]*\s+\S+', re.I)
_JWT_VALUE = re.compile(r'(?<![\w-])([A-Za-z0-9_-]{8,})\.([A-Za-z0-9_-]{2,})\.[A-Za-z0-9_-]+(?![\w-])')


def _credential_value(value: str) -> bool:
    """Recognize access syntax independently of the publisher's field names."""
    if _CREDENTIAL_VALUE.search(value) or re.fullmatch(r'\s*Bearer\s+\S+\s*', value, re.I):
        return True
    basic = re.fullmatch(r'\s*Basic\s+(\S+)\s*', value, re.I)
    if basic:
        try:
            if b':' in base64.b64decode(basic[1], validate=True):
                return True
        except ValueError:
            pass
    for match in _JWT_VALUE.finditer(value):
        try:
            header, payload = (json.loads(base64.urlsafe_b64decode(part + '=' * (-len(part) % 4)))
                               for part in match.group(1, 2))
        except (ValueError, UnicodeError):
            continue
        if isinstance(header, dict) and isinstance(payload, dict) and isinstance(header.get('alg'), str):
            return True
    # Public citation URLs remain intact, including citations embedded in prose.
    return any(_private_url(match[0].rstrip('.,);]'))
               for match in re.finditer(r'https?://[^\s<>"\']+', value, re.I))


def _private_access(value: str) -> bool:
    """Private hosts can be bare endpoints or scp-style user@host locations."""
    if not value or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit('//' + value)
        return bool(parsed.hostname and _private_host(parsed.hostname))
    except ValueError:
        return False


def _private_host(value: str) -> bool:
    token = value.strip().lower().rstrip('.')
    if not token or any(char.isspace() for char in token) or '/' in token:
        return False
    try:
        return not ipaddress.ip_address(token.strip('[]')).is_global
    except ValueError:
        pass
    host = token.rsplit(':', 1)[0] if re.search(r':\d+$', token) else token
    try:
        return not ipaddress.ip_address(host.strip('[]')).is_global
    except ValueError:
        pass
    return (host == 'localhost' or host.endswith(('.localhost', '.local', '.internal', '.lan', '.ts.net')))


def _private_url(value: str) -> bool:
    """Recognize access URLs without trying DNS or fetching any source."""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https'):
            return bool(parsed.scheme)
        host = (parsed.hostname or '').lower().rstrip('.')
        if not host or parsed.username is not None or parsed.password is not None:
            return True
        if '.' not in host or _private_host(host):
            return True
        try:
            if not ipaddress.ip_address(host).is_global:
                return True
        except ValueError:
            pass
        if host == 'hooks.slack.com' and parsed.path.startswith('/services/'):
            return True
        if host in ('discord.com', 'discordapp.com') and re.match(r'/api/(?:v\d+/)?webhooks/', parsed.path):
            return True
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True) + parse_qsl(parsed.fragment, keep_blank_values=True):
            name = _name(key)
            if _credential_key(key) or name in _URL_QUERY_SECRETS or name.startswith(('xamz', 'xgoog')):
                return True
    except ValueError:
        return True
    return False


def _location_value(value: str) -> str:
    return unquote(value.strip())


def _private_location(value: str) -> bool:
    value = _location_value(value)
    if value.lower().startswith(_ACCESS_URI_PREFIXES):
        return True
    if '://' in value or value.startswith('//'):
        return _private_url(value if not value.startswith('//') else 'https:' + value)
    windows = PureWindowsPath(value)
    return (PurePosixPath(value).is_absolute() or bool(windows.drive)
            or value.startswith(('~/', '~\\', '\\'))
            or '..' in value.replace('\\', '/').split('/'))


def _source_name(value: str) -> str:
    """A source file's basename is useful identity; its access directory is private."""
    value = _location_value(value)
    if '://' in value or value.startswith('//') or value.lower().startswith(_ACCESS_URI_PREFIXES):
        try:
            path = urlsplit(value).path
        except ValueError:
            return ''
        if not path.startswith('/'):
            return ''
        value = path
    return PureWindowsPath(value).name if '\\' in value else PurePosixPath(value).name


def _reader_attribute(key: str) -> str | None:
    """Decode the exact file/episode attribute names emitted by formats.h5_text."""
    for prefix in ('file attribute ', 'episode attribute '):
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def _location_field(field: str) -> bool:
    return field in _LOCATIONS or field.endswith(('url', 'uri', 'path', 'paths', 'urls', 'uris'))


def _whole_uri(value) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    if not token or any(char.isspace() for char in token):
        return False
    token = _location_value(token)
    return (bool(re.match(r'^[a-z][a-z0-9+.-]*://', token, re.I)) or token.startswith('//')
            or token.lower().startswith(_ACCESS_URI_PREFIXES))


def _filesystem_list(parts) -> bool:
    if not parts or any(_name(part) == 'fields' for part in parts):
        return False
    field = _name(_reader_attribute(parts[-1]) or parts[-1])
    return (field in ('sources', 'paths', 'files') or field.endswith('paths')
            or any(_name(_reader_attribute(part) or '') in _ATTRIBUTE_LOCATIONS for part in parts))


def location_list_receipt(path, values, limit):
    """Retain only the existing preview of a typed location list, never arbitrary arrays."""
    if not isinstance(values, list):
        return None
    preview = values if len(values) <= limit else values[:4]
    filesystem = _filesystem_list(path)
    private = any(isinstance(v, str) and (_whole_uri(v) or filesystem) and _private_location(v) for v in preview)
    return {'values': list(preview), 'count': len(values)} if private else None


def _table_error(value):
    if isinstance(value, str) and value.endswith(')'):
        for prefix in ('table read failed (', 'metadata table read failed ('):
            if value.startswith(prefix):
                return prefix[:-2] + '; original reader error retained privately'
    return value


def _partial_coverage(value: str) -> str:
    marker = 'it was not run on every camera, as the check stopped with an error ('
    position = value.lower().find(marker)
    if position >= 0 and value.rstrip('. ').endswith(')'):
        return value[:position] + 'It was not run on every camera; original camera reader details are retained privately.'
    return value


def _source_keys(value: dict) -> dict:
    reserved = {key for key in value if not _private_location(key)}
    used, keys = set(), {}
    for key in sorted(value):
        public = key
        if _private_location(key):
            base = _source_name(key) or 'source file'
            public = base
            if public in reserved or public in used:
                suffix = hashlib.sha256(key.encode()).hexdigest()
                public = base + '#' + suffix[:12]
                index = 1
                while public in reserved or public in used:
                    public = base + '#' + suffix + '.' + str(index)
                    index += 1
        keys[key] = public
        used.add(public)
    return keys


def reader_context_field(name: str, value):
    """Keep reader-owned source-map typing when an inspection unwraps its value."""
    if _name(name) in _READER_CONTEXT_MAPS:
        return project({name: value}).get(name)
    return value


def reader_annotation_projection(text, notes):
    """Project canonical serialized notes; unrelated publisher prose has no typed proof."""
    if not isinstance(text, str) or not isinstance(notes, (dict, list)):
        return None
    from prepare.formats import annotation_text
    try:
        for files in (False, True):
            if text == annotation_text(notes, files=files):
                return annotation_text(reader_context_field('uploader_notes', notes), files=files) or ''
    except (TypeError, ValueError):
        return None
    return None


def _inspection_source_name(value):
    descriptor = value.get('descriptor') if isinstance(value, dict) else None
    name = descriptor.get('name') if isinstance(descriptor, dict) else None
    return name if (isinstance(name, str) and descriptor.get('kind') in ('metadata', 'context')
                    and _name(name) in _READER_CONTEXT_MAPS and value.get('mode') == 'metadata') else None


def _source_pointer_keys(name, value, prefix=()):
    """Alias only source-owner positions; native recorded fields are separate keys."""
    if _name(name) == 'fields':
        return {}
    keys = {}
    if isinstance(value, dict):
        if _name(name) in _NOTE_SOURCE_MAPS:
            keys[prefix] = _source_keys(value)
        for key, child in value.items():
            if _name(key) != 'fields':
                keys.update(_source_pointer_keys(key, child, prefix + (key,)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.update(_source_pointer_keys('', child, prefix + (str(index),)))
    return keys


def _inspection_record(value: dict) -> dict:
    """Restore source-map ownership in saved inspection values and citations."""
    receipts = value.get('inspections')
    result = dict(value)
    if isinstance(value.get('parts'), list):
        result['parts'] = [{**part, 'record': _inspection_record(part['record'])}
                           if isinstance(part, dict) and isinstance(part.get('record'), dict) else part
                           for part in value['parts']]
    if not isinstance(receipts, list):
        return result

    def truncated_map(receipt):
        selected = receipt.get('value')
        return (isinstance(selected, dict) and selected.get('truncated') is True
                and isinstance(selected.get('preview'), str))

    aliases = {}
    for receipt in receipts:
        if (_inspection_source_name(receipt) and receipt.get('pointer') == ''
                and isinstance(receipt.get('value'), dict) and not truncated_map(receipt)):
            field = receipt['descriptor'].get('id') or receipt.get('field_id')
            if isinstance(field, str):
                aliases[field] = _source_pointer_keys(receipt['descriptor']['name'], receipt['value'])

    def pointer_alias(pointer, keys, name):
        if not isinstance(pointer, str) or not pointer.startswith('/'):
            return pointer
        prefix, public, native = (), [], _name(name) == 'fields'
        for encoded in pointer[1:].split('/'):
            token = encoded.replace('~1', '/').replace('~0', '~')
            owners = keys.get(prefix, {})
            source_owner = not native and _name(prefix[-1] if prefix else name) in _NOTE_SOURCE_MAPS
            if token in owners:
                public.append(owners[token].replace('~', '~0').replace('/', '~1'))
            elif source_owner and _private_location(token):
                return None
            else:
                public.append(encoded)
            native = native or _name(token) == 'fields'
            prefix += (token,)
        return '/' + '/'.join(public)

    references, shown = {}, []
    for receipt in receipts:
        if not _inspection_source_name(receipt):
            descriptor = receipt.get('descriptor') if isinstance(receipt, dict) else None
            if (isinstance(descriptor, dict) and descriptor.get('name') == 'uploader_annotation'
                    and descriptor.get('kind') in ('metadata', 'context') and receipt.get('mode') == 'metadata'):
                for original in receipts:
                    if (_inspection_source_name(original) == 'uploader_notes' and original.get('pointer') == ''):
                        projected = reader_annotation_projection(receipt.get('value'), original.get('value'))
                        if projected is not None:
                            receipt = {**receipt, 'value': projected}
                            break
            shown.append(receipt)
            continue
        receipt = dict(receipt)
        descriptor = dict(receipt['descriptor'])
        field = descriptor.get('id') or receipt.get('field_id')
        keys = aliases.get(field, {}) if isinstance(field, str) else {}
        details = descriptor.get('descriptor')
        if isinstance(details, dict) and isinstance(details.get('children'), list):
            children = details['children']
            if all(isinstance(child, str) for child in children):
                owners = keys.get((), {})
                descriptor['descriptor'] = {**details, 'children': [owners.get(child, child) for child in children]}
                if not owners and _name(descriptor['name']) in _NOTE_SOURCE_MAPS:
                    descriptor['descriptor'].update(children=list(_source_keys(dict.fromkeys(children)).values()),
                                                    children_executable=False)
        elif (isinstance(details, dict) and details.get('truncated') is True
              and isinstance(details.get('preview'), str)):
            descriptor['descriptor'] = {**details, 'preview': 'Original source-map descriptor preview is retained privately.'}
        receipt['descriptor'] = descriptor
        old = receipt.get('pointer')
        public = pointer_alias(old, keys, descriptor['name'])
        if isinstance(receipt.get('value'), (dict, list)):
            if truncated_map(receipt):
                receipt['value'] = {**receipt['value'], 'preview': 'Original source-map preview is retained privately.'}
                receipt['limitations'] = list(receipt.get('limitations') or [])
                note = 'The saved metadata preview lacks complete source-map structure for public owner aliases.'
                if note not in receipt['limitations']:
                    receipt['limitations'].append(note)
            else:
                tokens = [token.replace('~1', '/').replace('~0', '~') for token in old.split('/')[1:]] if isinstance(old, str) else []
                scope = tokens[-1] if tokens else descriptor['name']
                native_scope = receipt.get('value_scope') == 'native_fields' or any(_name(token) == 'fields' for token in tokens)
                if native_scope:
                    scope = 'fields'
                relative_keys = _source_pointer_keys(scope, receipt['value'])
                if not native_scope and _name(scope) in _READER_CONTEXT_MAPS:
                    receipt['value'] = reader_context_field(scope, receipt['value'])
                else:
                    # A source-owner selection retains note typing around its native members.
                    selected = {'fields': receipt['value']} if native_scope else receipt['value']
                    selected = project({'recorded_metadata': {'selection': selected}})['recorded_metadata']['selection']
                    receipt['value'] = selected['fields'] if native_scope else selected
                if 'exact_metadata_values' in receipt:
                    receipt['exact_metadata_values'] = [{**item, 'pointer': pointer_alias(item.get('pointer'), relative_keys, scope)}
                        for item in receipt.get('exact_metadata_values') or [] if isinstance(item, dict)]
        if public is None:
            receipt.pop('pointer', None)
            receipt['pointer_executable'] = False
            if isinstance(old, str) and 'fields' in [
                    _name(token.replace('~1', '/').replace('~0', '~')) for token in old.split('/')[1:]]:
                receipt['value_scope'] = 'native_fields'
            note = 'Original source-owner pointer is retained privately; this saved selection has no verified public alias.'
            receipt['limitations'] = list(receipt.get('limitations') or [])
            if note not in receipt['limitations']:
                receipt['limitations'].append(note)
        elif isinstance(old, str):
            receipt['pointer'] = public
        if isinstance(receipt.get('id'), str) and isinstance(old, str):
            references[(receipt['id'], old)] = public
        shown.append(receipt)
    result['inspections'] = shown

    def citations(item):
        if isinstance(item, list):
            return [citations(child) for child in item]
        if not isinstance(item, dict):
            return item
        item = {key: child if key in ('parts', 'inspections') else citations(child) for key, child in item.items()}
        key = (item.get('evidence_id'), item.get('pointer'))
        if all(isinstance(part, str) for part in key) and key in references:
            if references[key] is None:
                item.pop('pointer', None)
                item['pointer_executable'] = False
            else:
                item['pointer'] = references[key]
        return item
    return citations(result)


def _uploader_item(item: dict) -> dict:
    """Project the reader's flattened display schema using its original field path."""
    result = dict(item)
    parts = item.get('_uploader_field_path')
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        # Historical boards used this exact separator but retained no path array.
        parts = str(item.get('name') or '').split(' / ')
    shown = list(parts)
    for i, part in enumerate(parts):
        if any(_name(parent) == 'fields' for parent in parts[:i]):
            break
        if i and _name(parts[i - 1]) in _NOTE_SOURCE_MAPS and _private_location(part):
            shown[i] = _source_name(part) or 'source file'
    result['name'] = ' / '.join(shown) if shown else (item.get('name') or 'note')
    attributes = [_name(attribute) for i, part in enumerate(parts)
                  if not any(_name(parent) == 'fields' for parent in parts[:i])
                  and (attribute := _reader_attribute(part)) is not None]
    native = False
    for index, part in enumerate(parts):
        native = native or _name(part) == 'fields'
        parent = parts[index - 1] if index else ''
        if _credential_key(_reader_attribute(part) or part, native=native,
                           owner=_reader_attribute(parent) or parent):
            result['value'] = 'withheld'
            return result
    value = item.get('value')
    terminal = _name(parts[-1]) if parts else ''
    locations = item.get('_uploader_location_values')
    if not isinstance(locations, dict) and isinstance(value, str):
        summary = re.fullmatch(r'([0-9]+) values, (.+), \.\.\.', value)
        tokens = (summary[2] if summary else value).split(', ', 24)
        count = int(summary[1]) if summary else len(tokens)
        # The reader's exact display delimiter lets whole URI members retain
        # their type. Other members stay display strings, never guessed values.
        if (len(tokens) <= (4 if summary else 24) and any(_whole_uri(token) for token in tokens)
                and count >= len(tokens)):
            locations = {'values': tokens, 'count': count}
    if isinstance(locations, dict) and isinstance(locations.get('values'), list):
        filesystem = _filesystem_list(parts)
        projected = [_project(v, 'urls' if _whole_uri(v) or filesystem else '') for v in locations['values']]
        for i, original in enumerate(locations['values']):
            if (projected[i] is _DROP and isinstance(original, str)
                    and not _whole_uri(original)):
                projected[i] = _source_name(original) or 'source file'
        display = ', '.join('withheld' if v is _DROP else str(v) for v in projected)
        count = locations.get('count', len(projected))
        result['value'] = display if count == len(projected) else f'{count} values, {display}, ...'
        return result
    # A reader source record owns its immediate source property. The fields
    # below that record are the publisher's native schema, not access metadata.
    native_fields = any(_name(part) == 'fields' for part in parts[:-1])
    names = [_name(part) for part in parts]
    reader_claim = any(name in _NOTE_CLAIMS for name in names)
    reader_issue = any(name in _NOTE_SOURCE_MAPS for name in names) and 'issues' in names
    if not native_fields and (reader_claim or reader_issue) and terminal == 'why':
        result['value'] = value = _table_error(value)
    owner_property = (len(parts) >= 3 and _name(parts[-3]) in _NOTE_SOURCE_MAPS
                      and terminal == 'source')
    table_claim = (terminal == 'source' and 'recordedannotationtables' in [_name(p) for p in parts]
                   and 'claims' in [_name(p) for p in parts[:-1]])
    source_value = not native_fields and (owner_property or table_claim or terminal in _NOTE_SOURCE_VALUES
                                         or (terminal == 'source' and (reader_claim or reader_issue))
                                         or any(a in _ATTRIBUTE_LOCATIONS for a in attributes)
                                         or any(a not in ('source', 'path') and _location_field(a) for a in attributes)
                                         or (bool(attributes) and _reader_attribute(parts[-1]) is None
                                             and _location_field(terminal)))
    if isinstance(value, str) and source_value and _private_location(value):
        result.pop('value', None)
        result['value'] = _source_name(value) or 'source file'
    elif isinstance(value, str) and _credential_value(value):
        result['value'] = 'withheld'
    return result


def _uploader_group(value: dict) -> dict:
    if not isinstance(value.get('items'), list) or value.get('kind') not in ('notes', 'row'):
        return value
    result = dict(value)
    result['items'] = [_uploader_item(item) if isinstance(item, dict) else item for item in value['items']]
    title = value.get('title')
    if isinstance(title, str) and title.startswith('Row of ') and _private_location(title[7:]):
        result['title'] = 'Row of ' + (_source_name(title[7:]) or 'source file')
    return result


def reader_failure_what(kind: str) -> str | None:
    """Public availability wording for typed reader failures, including historical ones."""
    failures = {
        'context_unreadable': 'Recorded episode details could not be read.',
        'depth_not_decodable': 'The recorded depth video does not decode; the page shows no depth for this camera.',
        'depth_clip_failed': 'Depth could not be drawn for the board; the page shows no depth for this camera.',
        'model_no_reply': 'No verified model reply was recorded; footage, checks and sensors remain available.',
        'model_reply_not_shown': 'The model reply could not be read by the board; its original content is retained.',
        'label_output_unreadable': 'The labelling output file could not be read; footage, checks and sensors remain available.',
        'no_part_labelled': 'No part of this recording has labels; footage, checks and sensors remain available.',
        'part_not_labelled': 'This recording part has no labels; labels come from the other parts.',
    }
    return failures.get(kind) if isinstance(kind, str) else None


def _reader_issue(value: dict) -> dict:
    """Only machine reader diagnostics receive source-location migration."""
    result = dict(value)
    kind = value.get('kind')
    safe = value.get('_reader_public_what')
    if isinstance(safe, str):
        result['what'] = safe
    elif kind != 'context_unreadable' and reader_failure_what(kind):
        result['what'] = reader_failure_what(kind)
    if kind == 'context_unreadable' and 'error' in result:
        result['error'] = value.get('_reader_public_error') or reader_failure_what(kind)
    what = result.get('what')
    source = value.get('source')
    if isinstance(what, str) and isinstance(source, str) and _private_location(source):
        result['what'] = what.replace(source, _source_name(source) or 'source file')
    elif (value.get('kind') == 'metadata_limit' and isinstance(what, str)
          and what.endswith(_METADATA_LIMIT_SUFFIX)):
        source = what[:-len(_METADATA_LIMIT_SUFFIX)]
        if _private_location(source):
            result['what'] = (_source_name(source) or 'Recorded annotation data') + _METADATA_LIMIT_SUFFIX
    elif value.get('kind') == 'metadata_unreadable' and isinstance(what, str):
        # Historical annotation diagnostics retained the source in their exact generated sentence.
        if what.endswith(_ANNOTATION_CODES_SUFFIX):
            column, separator, source = what[:-len(_ANNOTATION_CODES_SUFFIX)].rpartition(' in ')
            if separator and _private_location(source):
                result['what'] = column + separator + (_source_name(source) or 'source file') + _ANNOTATION_CODES_SUFFIX
    what = result.get('what')
    if kind == 'metadata_unreadable' and isinstance(what, str):
        for marker in (' could not be fully read: ', ' retains an unassigned metadata claim: '):
            head, separator, detail = what.partition(marker)
            if separator:
                public_detail = _table_error(detail[:-1] if detail.endswith('.') else detail)
                if public_detail != (detail[:-1] if detail.endswith('.') else detail):
                    if marker.startswith(' retains') and _private_location(head):
                        head = _source_name(head) or 'source file'
                    result['what'] = head + separator + public_detail + '.'
    return result


def _part_failures(parts: list) -> list:
    """Typed machine failure spans, with operational diagnostics retained privately."""
    public = []
    for part in parts:
        if not isinstance(part, dict):
            public.append(part)
            continue
        row = dict(part)
        if row.get('final_dispatch_outcome') not in ('claimed', 'unverified', 'not dispatched'):
            row.pop('final_dispatch_outcome', None)
        if row.get('no_reply_kind') != 'spend_cap' or row.get('final_dispatch_outcome') != 'not dispatched':
            row.pop('no_reply_kind', None)
        if row.get('finish_reason') != 'length':
            row.pop('finish_reason', None)
        if 'why' in row:
            row['why'] = 'No verified labels were recorded for this part.'
        public.append(row)
    return public


def _label_failure(value: dict) -> dict:
    result = dict(value)
    for key, allowed in (('final_dispatch_outcome', ('claimed', 'unverified', 'not dispatched')),
                         ('no_reply_kind', ('spend_cap',))):
        if value.get('status') != 'no_reply' or value.get(key) not in allowed:
            result.pop(key, None)
    if value.get('status') in ('unreadable', 'not_shown') and 'error' in value:
        result['error'] = 'The recorded model output could not be read; original reader details are retained privately.'
    elif value.get('status') == 'no_reply' and 'why' in value:
        result['why'] = 'No verified model reply was recorded.'
    elif value.get('status') == 'no_part' and isinstance(value.get('parts'), list):
        result['parts'] = _part_failures(value['parts'])
    return result


def _private_binding_pointer(value) -> bool:
    if not isinstance(value, str):
        return False
    parts = [part.replace('~1', '/').replace('~0', '~') for part in value.split('/')]
    for i, part in enumerate(parts[:-1]):
        if _name(part) == 'fields':
            break
        if (_name(part) in _NOTE_SOURCE_MAPS
                and _private_location(parts[i + 1])):
            return True
    return False


def _credential_binding(binding: dict) -> bool:
    # Array bindings identify recorded channels; an exact native key named token
    # is not a context credential attribute.
    if binding.get('file') and binding.get('key') is not None:
        return False
    value = binding.get('context_path')
    if not isinstance(value, str):
        return False
    parts = [part.replace('~1', '/').replace('~0', '~') for part in value.split('/')]
    native = False
    for i, part in enumerate(parts):
        native = native or _name(part) == 'fields'
        if i and _name(parts[i - 1]) in _NOTE_SOURCE_MAPS:
            continue
        parent = parts[i - 1] if i else ''
        if _credential_key(_reader_attribute(part) or part, native=native,
                           owner=_reader_attribute(parent) or parent):
            return True
    return False


def _dictionary_values(value: dict) -> dict:
    fields = value.get('fields')
    if not isinstance(fields, list):
        return value
    secret_ids = {field['id'] for field in fields if isinstance(field, dict) and isinstance(field.get('id'), str)
                  and any(_credential_binding(binding)
                          for binding in field.get('bindings') or [] if isinstance(binding, dict))}
    if not secret_ids:
        return value
    channels = {'summary', 'value', 'values', 'preview', 'minimum', 'maximum', 'mean',
                'standard_deviation', 'finite_fraction'}
    result = {**value, 'fields': [{key: item for key, item in field.items() if key not in channels}
              if isinstance(field, dict) and field.get('id') in secret_ids else field for field in fields]}
    for key in ('entries', 'machine_entries'):
        if isinstance(value.get(key), dict):
            result[key] = {ident: entry for ident, entry in value[key].items() if ident not in secret_ids}
    return result


def _project(value, field: str = '', attribute_owner: str = '', native_fields: bool = False,
             note_owner: str = '', machine_checks: bool = False):
    native_fields = native_fields or field in ('fields', 'publisherlabels')
    note_owner = field if field in _NOTE_SOURCE_MAPS | _NOTE_CLAIMS else note_owner
    machine_checks = (machine_checks or field in ('datasetchecks', 'setasidechecks')) and field != 'readerissues' and not native_fields
    if isinstance(value, dict):
        if not native_fields:
            value = _dictionary_values(value)
            if field == 'evidenceinspection':
                value = _inspection_record(value)
        if field == 'uploadernotes' and not native_fields:
            value = _uploader_group(value)
        elif field == 'readerissues' and not native_fields:
            value = _reader_issue(value)
        if machine_checks:
            status = value.get('status')
            errors = [key for key in ('error', 'decode_error') if isinstance(value.get(key), str)]
            if errors:
                value = {**value, **{key: 'Original reader error details are retained privately.' for key in errors}}
            if status in ('errored', 'error') and 'why' in value:
                value = {**value, 'why': 'The check could not complete; original reader details are retained privately.'}
            elif status is None and errors and isinstance(value.get('why'), str):
                why = value['why']
                if why.lower().startswith('the check stopped with an error (') and why.rstrip('. ').endswith(')'):
                    value = {**value, 'why': 'The check could not complete; original reader details are retained privately.'}
            if value.get('check') == 'video_decode_failure' and isinstance(value.get('evidence'), str):
                camera, separator, _error = value['evidence'].partition(' fails to decode: ')
                if separator and camera.startswith('camera '):
                    value = {**value, 'evidence': camera + ' fails to decode; original camera reader details are retained privately.'}
        result = {}
        source_keys = _source_keys(value) if field in _NOTE_SOURCE_MAPS and not native_fields else {}
        for key, item in value.items():
            if field == 'bindings' and _name(key) == 'contextpath' and _private_binding_pointer(item):
                continue
            if key == '_label_failed' and isinstance(item, dict) and not native_fields:
                item = _label_failure(item)
            if key == '_stitched' and isinstance(item, dict) and not native_fields and isinstance(item.get('missing'), list):
                item = {**item, 'missing': _part_failures(item['missing'])}
            name = _name(key)
            attribute = _reader_attribute(key) if not native_fields else None
            owner = _name(attribute) if attribute is not None else attribute_owner
            if (key in PRIVATE_KEYS or _credential_key(key, native=native_fields, owner=attribute_owner or field)
                    or name in _PRIVATE_LOCATIONS
                    or (attribute is not None and _credential_key(attribute))
                    or (field == 'compare' and name in ('runid', 'code'))):
                continue
            item_native, item_notes = native_fields, note_owner
            if key == 'value' and (source_name := _inspection_source_name(value)):
                item_notes = _name(source_name)
                # A selected metadata subtree retains its original native-field scope.
                pointer = value.get('pointer')
                if isinstance(pointer, str):
                    item_native = item_native or 'fields' in [
                        _name(token.replace('~1', '/').replace('~0', '~')) for token in pointer.split('/')[1:]]
                item_native = item_native or value.get('value_scope') == 'native_fields'
            shown = _project(item, name, owner, item_native, item_notes, machine_checks)
            if shown is not _DROP:
                result[source_keys.get(key, key)] = shown
        return result
    if isinstance(value, (list, tuple)):
        # Do not move sensor sample positions when a typed string is withheld.
        return [None if (shown := _project(item, field, attribute_owner, native_fields, note_owner, machine_checks))
                is _DROP else shown for item in value]
    if isinstance(value, str):
        if machine_checks and field in ('why', 'unchecked'):
            value = _partial_coverage(value)
        if note_owner and not native_fields and field == 'source' and _private_location(value):
            return _source_name(value) or 'source file'
        if note_owner and not native_fields and field == 'why':
            value = _table_error(value)
        if attribute_owner in _ATTRIBUTE_LOCATIONS and _private_location(value):
            return _source_name(value) or 'source file'
        # An entire URL is a typed value even in an extension field. Do not scan
        # or rewrite URL fragments embedded in publisher prose.
        stripped = value.strip()
        version = field.endswith('version') and bool(re.fullmatch(r'\d+(?:\.\d+)+', stripped))
        if _credential_value(value) or (not version and _private_access(stripped)):
            return _DROP
        if _whole_uri(stripped) and _private_location(stripped):
            return _DROP
        # Direct source/path attributes can name HDF members; nested access fields retain protection.
        direct_attribute = attribute_owner in ('source', 'path') and field in (
            'fileattribute' + attribute_owner, 'episodeattribute' + attribute_owner)
        location = not direct_attribute and not (native_fields and note_owner) and _location_field(field)
        # context_path is a JSON pointer; it is not a filesystem path.
        if field == 'contextpath':
            location = False
        if location and _private_location(value):
            return _DROP
    return value


def project(value):
    """Return independent public JSON; omit private fields, use null in arrays.

    Values must already be JSON-compatible. This function does not read, resolve,
    write or mutate inputs. Run evidence freshness and human overrides first.
    Native HDF keys and JSON pointers remain meaningful public identifiers.
    """
    result = _project(value)
    return None if result is _DROP else result


def public_build_manifest(build: dict) -> dict:
    """Publish build identity and availability, without operational directories.

    Unknown build fields fail closed. Detailed skip diagnostics remain in the
    original private receipt because exception messages can contain local paths.
    """
    result = {key: build[key] for key in (
        'build_id', 'built_at', 'code', 'enc', 'public_base', 'episodes', 'datasets',
        'media', 'missing_media',
    ) if key in build}
    for key in ('compare', 'hands', 'keypoints', 'sensors'):
        value = build.get(key)
        if isinstance(value, dict):
            result[key] = {k: value[k] for k in ('files', 'bytes') if k in value}
            if isinstance(value.get('skipped'), dict):
                result[key]['skipped'] = {name: 'unavailable' for name in value['skipped']}
        elif key in build:
            result[key] = None
    return project(result)
