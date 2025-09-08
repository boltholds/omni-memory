{% set ctx = context_text %}
{% set style_line = style_hint %}
{% set lang_line  = lang_hint %}

Question:
{{ q }}

Context:
{{ ctx }}

Instructions: {{ style_line }} {{ lang_line }}
