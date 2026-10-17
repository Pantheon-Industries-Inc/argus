"""Provider-published ceilings for capped OpenRouter requests without local list rates."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
import json
import math
import urllib.parse
import urllib.request


def quote(model: str, timeout: int) -> dict:
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
        endpoints = [endpoint for endpoint in endpoints if endpoint.get('service_tier') in (None, 'standard', 'default')]
        if not endpoints:
            raise ValueError('standard endpoint pricing is missing')
        # Claude web search requires an explicitly supplied search tool.
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        # Model variants such as :online can enable search outside that contract.
        claude = model.startswith('anthropic/claude-') and ':' not in model
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
                                    'internal_reasoning', 'discount', 'web_search') and rate:
                        raise ValueError('unsupported endpoint charge: ' + name)
                    rates[name] = max(rates.get(name, 0), rate)
            if rates.get('web_search', 0) and not claude:
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
        return {'model': model, 'source': url, 'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'prompt_token_bound': context, 'input_usd_per_token': input_rate,
                'output_usd_per_token': output_rate, 'request_usd': maxima['request'],
                'image_usd': maxima['image'],
                'bound_basis': ('Full published context window; separate image and request fees bounded by provider.max_price. '
                                + ('Claude search requires an explicit tool; none is supplied. Cache TTL is 5 minutes.'
                                   if claude else 'No published native search charge.')),
                'provider_max_price': {**maxima, 'prompt': float(Decimal(str(maxima['prompt'])) * 1_000_000),
                                       'completion': float(Decimal(str(maxima['completion'])) * 1_000_000)}}
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f'No verified pricing rates for {model}: {error}; no paid request was dispatched') from error


def cost_bound(content: list, max_tokens: int, quote: dict) -> float:
    """Use the full published context bound because image tokenizers differ by provider."""
    images = sum(part.get('type') == 'image_url' for part in content)
    amount = (quote['prompt_token_bound'] * quote['input_usd_per_token']
              + max_tokens * quote['output_usd_per_token'] + quote['request_usd'] + images * quote['image_usd'])
    if not math.isfinite(amount) or amount < 0:
        raise ValueError('provider cost bound is invalid')
    return math.ceil(amount * 1_000_000) / 1_000_000
