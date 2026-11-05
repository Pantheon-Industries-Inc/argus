"""Provider-published ceilings for capped OpenRouter requests without local list rates."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
import json
import math
import urllib.parse
import urllib.request


# Prices published for inputs a request here never carries: audio, and its cached tokens (Gemini lists both). They are
# accepted in a quote because cost_bound refuses any part but text and images, so they cannot be charged.
UNSENT_INPUT_CHARGES = ('audio', 'input_audio_cache')
# The request parts cost_bound bounds; any other (an audio or file part) has no verified bound under a quote
BOUNDED_PARTS = ('text', 'image_url')


# Set by the first reply whose usage counts a web search: the fee applies after all, so call_model sends nothing more.
search_seen = None


def searches_counted(response: dict) -> int:
    """How many web searches a reply's usage reports (https://openrouter.ai/docs/guides/features/server-tools/web-search)."""
    usage = response.get('usage') if isinstance(response, dict) else None
    details = usage.get('server_tool_use_details') if isinstance(usage, dict) else None
    count = details.get('web_search_requests') if isinstance(details, dict) else None
    return count if type(count) is int and count > 0 else 0


def quote(model: str, timeout: int, *, allow_unverified_fees: bool = False, service_tier: str | None = None) -> dict:
    # https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model
    # https://openrouter.ai/docs/guides/routing/provider-selection#max-price
    # https://openrouter.ai/docs/guides/overview/models#pricing-overrides
    url = 'https://openrouter.ai/api/v1/models/' + urllib.parse.quote(model, safe='/') + '/endpoints'
    # Catalog metadata is public. A revoked paid key must not block this free lookup.
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 30)) as response:
            data = json.loads(response.read())['data']
        endpoints = data['endpoints']
        if data['id'] != model or not isinstance(endpoints, list) or not endpoints:
            raise ValueError('model identity or endpoint list is missing')
        if any(not isinstance(endpoint, dict) for endpoint in endpoints):
            raise ValueError('endpoint metadata is invalid')
        # Requests use the standard service tier. The resulting max_price also prevents
        # fallback to a more expensive premium endpoint.
        # A flex run is priced by the flex endpoints only, which also makes max_price refuse any dearer one.
        tiers = (service_tier,) if service_tier else (None, 'standard', 'default')
        endpoints = [endpoint for endpoint in endpoints if endpoint.get('service_tier') in tiers]
        if not endpoints:
            raise ValueError(f"{service_tier or 'standard'} endpoint pricing is missing")
        # Claude web search requires an explicitly supplied search tool.
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        # Model variants such as :online can enable search outside that contract.
        claude = model.startswith('anthropic/claude-') and ':' not in model
        # Gemini searches only when a request asks for it as well: on OpenRouter, through the web plugin or an :online
        # variant (https://openrouter.ai/docs/guides/features/web-search). Requests here supply neither, so a published
        # search price cannot apply to them.
        # OpenAI's models work the same way, apart from the ones built to search. OpenRouter publishes a search price on
        # them all, and every request here leaves search off, so a reply that counts one stops the run (call_model).
        openai = (model.startswith('openai/') and ':' not in model
                  and not any(word in model for word in ('search', 'deep-research')))
        searches_on_request = claude or openai or (model.startswith('google/gemini-') and ':' not in model)
        search_usd = 0.0
        maxima = dict.fromkeys(('prompt', 'completion', 'request', 'image'), 0.0)
        context = 0
        input_rate = output_rate = 0.0
        for endpoint in endpoints:
            tokens = endpoint['context_length']
            if type(tokens) is not int or tokens <= 0:
                raise ValueError('endpoint context bound is missing')
            context = max(context, tokens)
            pricing = endpoint['pricing']
            if not isinstance(pricing, dict) or not {'prompt', 'completion'} <= pricing.keys():
                raise ValueError('endpoint token pricing is missing')
            overrides = pricing.get('overrides', [])
            if not isinstance(overrides, list) or any(not isinstance(row, dict) for row in overrides):
                raise ValueError('endpoint pricing overrides are invalid')
            variants = [{name: value for name, value in pricing.items() if name != 'overrides'}]
            conditions = {'utc_start', 'utc_end', 'utc_days', 'min_prompt_tokens'}
            variants += [{**variants[0], **{name: value for name, value in row.items() if name not in conditions}}
                         for row in overrides]
            rates = {}
            # Bound every published context tier and time window, including ones not active now.
            for variant in variants:
                for name, value in variant.items():
                    if type(value) not in (str, int, float):
                        raise ValueError('unsupported pricing requires a verified bound')
                    rate = float(value)
                    if not math.isfinite(rate) or rate < 0:
                        raise ValueError('endpoint price must be finite and nonnegative')
                    if name not in (*maxima, 'input_cache_read', 'input_cache_write', 'input_cache_write_1h',
                                    'internal_reasoning', 'discount', 'web_search', *UNSENT_INPUT_CHARGES) and rate:
                        raise ValueError('unsupported endpoint charge: ' + name)
                    rates[name] = max(rates.get(name, 0), rate)
            search_usd = max(search_usd, rates.get('web_search', 0))
            if rates.get('web_search', 0) and not searches_on_request and not allow_unverified_fees:
                # max_price has no search dimension, so a snapshot search rate cannot
                # enforce a native-search ceiling. Do not authorize that capped call.
                raise ValueError('native search fees have no verified enforceable ceiling')
            for name in maxima:
                # A fee absent from metadata is forbidden by an explicit zero max_price.
                maxima[name] = max(maxima[name], rates.get(name, 0))
            # Requests set only a default 5-minute cache breakpoint for Claude.
            # https://openrouter.ai/docs/guides/best-practices/prompt-caching#cache-ttl-options
            cache_1h = 0 if claude else rates.get('input_cache_write_1h', 0)
            input_rate = max(input_rate, rates['prompt'], rates.get('input_cache_read', 0),
                             rates.get('input_cache_write', 0), cache_1h)
            # Reserve both if a provider reports a separate reasoning-token charge.
            output_rate = max(output_rate, rates['completion'] + rates.get('internal_reasoning', 0))
        if search_usd and not searches_on_request:
            print(f'WARNING {model}: unverified fee: a web search costs ${search_usd} and no cap limits it '
                  '(--allow-unverified-fees). The run stops at the first reply that counts one.', flush=True)
        return {**({'published_search_usd': search_usd} if search_usd else {}), 'model': model, 'source': url, 'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'prompt_token_bound': context, 'input_usd_per_token': input_rate,
                'output_usd_per_token': output_rate, 'request_usd': maxima['request'],
                'image_usd': maxima['image'],
                'bound_basis': ('Prompt bounded by the request text and, per image, the most a published tokenizer charges for its size, at most the published context window; separate image and request fees bounded by provider.max_price. '
                                + ('Claude search requires an explicit tool; none is supplied. Cache TTL is 5 minutes.'
                                   if claude else 'OpenAI or Gemini search requires the web plugin, a tool or :online; none is supplied.'
                                   if searches_on_request else 'Search fee not bounded: --allow-unverified-fees.' if search_usd
                                   else 'No published native search charge.')
                                + ' Audio prices apply only to audio input; requests carry text and images only.'),
                'provider_max_price': {**maxima, 'prompt': float(Decimal(str(maxima['prompt'])) * 1_000_000),
                                       'completion': float(Decimal(str(maxima['completion'])) * 1_000_000)}}
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f'No verified pricing rates for {model}: {error}; no paid request was dispatched') from error


def image_token_bound(width: int, height: int) -> int:
    """The most a published image tokenizer charges for one image of this size. gpt-4o mini's high detail rule charges
    the most per tile, 2,833 tokens plus 5,667 for each 512 px tile once the image is scaled down to fit 2048 px and
    its shorter side to 768 px. https://developers.openai.com/api/docs/guides/images-vision#tile-based-image-tokenization
    A fixed ceiling per image had been below it, so a request that fit a cap by the ceiling could still exceed it."""
    w, h = float(width), float(height)
    if max(w, h) > 2048:
        w, h = w * 2048 / max(w, h), h * 2048 / max(w, h)
    if min(w, h) > 768:
        w, h = w * 768 / min(w, h), h * 768 / min(w, h)
    return 2833 + 5667 * math.ceil(w / 512) * math.ceil(h / 512)


# The size that tiles the most once scaled (2048 px by 768 px: 4 by 2 tiles), for an image whose size is unknown
LARGEST_CHARGED_SIZE = (2048, 768)


def image_sizes(content: list) -> list:
    """(width, height) of every image part of a request, read from its own bytes. An image whose size cannot be read
    counts at LARGEST_CHARGED_SIZE, so an unknown is bounded by the most any image can cost, never left out."""
    import base64
    import io
    from PIL import Image
    sizes = []
    for part in content:
        if part.get('type') == 'image_url':
            try:
                raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                with Image.open(io.BytesIO(raw)) as image:
                    sizes.append(image.size)
            except Exception:  # noqa: BLE001 - an image whose size does not read is priced at the largest size
                sizes.append(LARGEST_CHARGED_SIZE)
    return sizes


def cost_bound(content: list, max_tokens: int, quote: dict) -> float:
    """Bound the prompt from the request itself: its text (a token is at least one byte) and, for each image, the
    most a published tokenizer charges for its size (image_token_bound), never more than the published context
    window. Reserving the whole window held about $6 for a request that costs well under $1, so a small cap
    labelled nothing."""
    unbounded = sorted({str(part.get('type')) for part in content if part.get('type') not in BOUNDED_PARTS})
    if unbounded:
        raise ValueError(f"a request part has no verified bound: {', '.join(unbounded)}")
    sizes = image_sizes(content)
    images = len(sizes)
    text_bytes = sum(len(str(part.get('text', '')).encode('utf-8')) for part in content if part.get('type') == 'text')
    prompt_tokens = min(quote['prompt_token_bound'],
                        text_bytes + sum(image_token_bound(*size) for size in sizes) + 1024)
    amount = (prompt_tokens * quote['input_usd_per_token']
              + max_tokens * quote['output_usd_per_token'] + quote['request_usd'] + images * quote['image_usd'])
    if not math.isfinite(amount) or amount < 0:
        raise ValueError('provider cost bound is invalid')
    return math.ceil(amount * 1_000_000) / 1_000_000
