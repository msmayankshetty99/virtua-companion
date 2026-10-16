import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {imageSource, normalizeLatex} from './format_text.mjs';
import {mediaURL} from './api.mjs';

// A remote image is a click-to-open link (opened in the browser by main's window-open handler), never an <img> that loads it;
// a local one comes through /api/media (imageSource in format_text.mjs). index.html's CSP refuses other image origins too.
function MarkdownImage({src, alt}) {
  const source = imageSource(src);
  if (source.kind === 'media') return <img alt={alt} src={mediaURL(source.path)}/>;
  if (source.kind === 'link') return <a className="image-link" href={source.href} target="_blank" rel="noreferrer" title={source.href}>{alt || 'Image'} ({source.host})</a>;
  return alt ? <span className="image-alt">{alt}</span> : null;
}

export default function FormattedText({text = ''}) {
  const normalized = normalizeLatex(text);
  return <div className="formatted-text"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, {strict: false, throwOnError: false}]]} components={{a: ({children, href}) => <a href={href} target="_blank" rel="noreferrer">{children}</a>, img: MarkdownImage}}>{normalized}</ReactMarkdown></div>;
}
