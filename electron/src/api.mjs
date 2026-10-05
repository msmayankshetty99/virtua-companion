export const API = 'http://127.0.0.1:8765';
export const EVENTS = 'ws://127.0.0.1:8765/ws/events';
export const mediaURL = path => API + '/api/media?path=' + encodeURIComponent(path);

// Electron main adds the API token to every backend request; renderer code never handles it.
export async function request(path, {method = 'GET', body, signal} = {}, confirmation) {
  const headers = {...(body === undefined ? {} : {'Content-Type': 'application/json'}), ...(confirmation ? {'X-Riko-Confirmation': confirmation} : {})};
  const response = await fetch(API + path, {
    method,
    ...(signal?{signal}:{}),
    headers,
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  });
  if (response.status === 428 && !confirmation) {
    // A security-sensitive change: Electron main asks the user and signs the exact change if they allow it.
    let challenge; try {challenge=(await response.clone().json()).detail?.confirm;} catch {}
    if (challenge) {
      const signature = await globalThis.riko?.confirmChange?.(challenge);
      if (!signature) throw new Error('Change cancelled. Nothing was saved.');
      return request(path, {method, body, signal}, signature);
    }
  }
  if (!response.ok) {
    const text=await response.text();
    let message=text;
    try {const value=JSON.parse(text);if(typeof value.detail==='string')message=value.detail;else if(typeof value.detail?.detail==='string')message=value.detail.detail;}catch{}
    if(response.status===404&&path.startsWith('/api/discord/'))message='This Python backend does not support the current Discord controls. Restart Python from this updated checkout, then retry.';
    throw new Error(message);
  }
  return response.json();
}

export function reportSurface(surface, command_id, status, error = '', bounds) {
  return request('/api/surfaces/result', {method: 'POST', body: {surface, command_id, status, error, bounds}}).catch(() => {});
}
