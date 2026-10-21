"""Per-channel protobuf decoding for the published dataset adapters."""
from __future__ import annotations

import json

from prepare import formats


class McapSourceError(ValueError):
    """A recorded channel lacks the schema or fields required by its adapter."""


def _check_fields(decoded, required: dict[str, str]) -> None:
    from google.protobuf.descriptor import FieldDescriptor as Field
    integers = {Field.TYPE_INT32, Field.TYPE_INT64, Field.TYPE_UINT32, Field.TYPE_UINT64,
                Field.TYPE_SINT32, Field.TYPE_SINT64, Field.TYPE_FIXED32, Field.TYPE_FIXED64,
                Field.TYPE_SFIXED32, Field.TYPE_SFIXED64}
    kinds = {"number": integers | {Field.TYPE_FLOAT, Field.TYPE_DOUBLE}, "integer": integers,
             "bytes": {Field.TYPE_BYTES}, "text": {Field.TYPE_STRING}, "bool": {Field.TYPE_BOOL},
             "message": {Field.TYPE_MESSAGE}}
    for path, kind in required.items():
        descriptor = getattr(decoded, 'DESCRIPTOR', None)
        for name in path.split('.'):
            if descriptor is None or name not in descriptor.fields_by_name:
                raise McapSourceError(f'Recorded schema lacks required field {path}.')
            field = descriptor.fields_by_name[name]
            descriptor = field.message_type
        if field.type not in kinds[kind.removesuffix('[]')] or field.is_repeated != kind.endswith('[]'):
            raise McapSourceError(f'Recorded field {path} must be {kind}.')


def decoded_messages(reader, fields: dict[str, dict[str, str]], failures: dict[str, str]):
    """Decode raw messages independently; one bad source invalidates only its own topic.

    Callers discard any earlier values on failed topics rather than joining a partial camera or state stream.
    Reader/container failures and unexpected decoder implementation errors propagate.
    """
    from google.protobuf.message import DecodeError
    from mcap_protobuf.decoder import DecoderFactory
    factories, decoders = [DecoderFactory()], {}
    from prepare.isolation import beating
    for schema, channel, message in beating(reader.iter_messages(topics=list(fields))):
        if channel.topic in failures:
            continue
        try:
            if channel.id not in decoders:
                decoder = formats._decoder_for(channel.message_encoding, schema, factories)
                if decoder is None:
                    raise McapSourceError(f'No decoder for {channel.message_encoding} with '
                                          f'{"no schema" if schema is None else schema.name}.')
                decoders[channel.id] = decoder
            decoded = decoders[channel.id](message.data)
            _check_fields(decoded, fields[channel.topic])
        except (formats.McapSchemaError, McapSourceError, DecodeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            failures[channel.topic] = formats.plain_error(error)
            continue
        yield schema, channel, message, decoded
