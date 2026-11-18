const API_BASE = '';

let conversationId = null;
let isWaiting = false;
let docCount = 0;

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const documentsList = document.getElementById('documents-list');
const uploadProgress = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const toastContainer = document.getElementById('toast-container');
const sidebar = document.getElementById('sidebar');
const sidebarOpen = document.getElementById('sidebar-open');
const sidebarClose = document.getElementById('sidebar-close');
const backdrop = document.getElementById('backdrop');
const statusText = document.getElementById('status-text');
const docCountBadge = document.getElementById('doc-count');

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3800);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

function normalizeMarkup(text) {
    return (text ?? '')
        .replace(/<\s*br\s*\/?\s*>/gi, '\n')
        .replace(/<\s*p\s*[^>]*>/gi, '')
        .replace(/<\/\s*p\s*>/gi, '\n\n')
        .replace(/<\s*\/?\s*(strong|b)\s*>/gi, '**')
        .replace(/<\s*\/?\s*(em|i)\s*>/gi, '*')
        .replace(/<\s*li\s*[^>]*>/gi, '\n- ')
        .replace(/<\s*(ul|ol|div|tr|h[1-6]|blockquote)\s*[^>]*>/gi, '')
        .replace(/<\/\s*(ul|ol|div|tr|h[1-6]|blockquote)\s*>/gi, '\n')
        .replace(/<[^>]+>/g, '');
}

function renderTables(html) {
    const lines = html.split('\n');
    const isRow = (line) => line.includes('|') && line.trim().length > 0;
    const isSeparator = (line) => /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.includes('-');
    const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim());
    const out = [];
    let i = 0;
    while (i < lines.length) {
        if (isRow(lines[i]) && i + 1 < lines.length && isSeparator(lines[i + 1])) {
            const head = cells(lines[i]);
            i += 2;
            const rows = [];
            while (i < lines.length && isRow(lines[i]) && !isSeparator(lines[i])) {
                rows.push(cells(lines[i]));
                i++;
            }
            out.push(
                '<table class="md-table"><thead><tr>' +
                head.map((cell) => `<th>${cell}</th>`).join('') +
                '</tr></thead><tbody>' +
                rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('') +
                '</tbody></table>'
            );
        } else {
            out.push(lines[i]);
            i++;
        }
    }
    return out.join('\n');
}

function formatAssistantText(text) {
    let html = escapeHtml(normalizeMarkup(text));
    html = html.replace(/&lt;br\s*\/?&gt;/gi, '\n');
    html = renderTables(html);
    html = html.replace(/^#{1,6}\s+(.*)$/gm, '<strong>$1</strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^\*])\*(?!\*)(.+?)\*(?!\*)/g, '$1<em>$2</em>');
    html = html.replace(/(^|[^_])_(?!_)(.+?)_(?!_)/g, '$1<em>$2</em>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
}

function openSidebar() { sidebar.classList.add('open'); backdrop.classList.add('show'); }
function closeSidebar() { sidebar.classList.remove('open'); backdrop.classList.remove('show'); }
sidebarOpen?.addEventListener('click', openSidebar);
sidebarClose?.addEventListener('click', closeSidebar);
backdrop?.addEventListener('click', closeSidebar);

function setStatus() {
    if (docCount > 0) {
        statusText.textContent = `${docCount} document${docCount > 1 ? 's' : ''} loaded. Ask a question below.`;
    } else {
        statusText.textContent = 'Upload a PDF to get started.';
    }
}

// Upload
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
['dragenter', 'dragover'].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) uploadDocument(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) { uploadDocument(fileInput.files[0]); fileInput.value = ''; } });

async function uploadDocument(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('Please upload a PDF file.', 'warning');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    uploadProgress.classList.add('active');
    progressFill.style.width = '15%';
    progressText.textContent = `Uploading ${file.name}...`;

    let fake = 15;
    const timer = setInterval(() => { fake = Math.min(fake + 8, 90); progressFill.style.width = `${fake}%`; }, 250);
    try {
        progressText.textContent = `Processing ${file.name}...`;
        const res = await fetch(`${API_BASE}/documents/upload`, { method: 'POST', body: formData });
        clearInterval(timer);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${res.status})`);
        }
        const data = await res.json();
        progressFill.style.width = '100%';
        progressText.textContent = 'Done';
        showToast(`Uploaded: ${data.document.name}`, 'success');
        setTimeout(() => uploadProgress.classList.remove('active'), 800);
        await loadDocuments();
        if (window.innerWidth <= 900) closeSidebar();
    } catch (err) {
        clearInterval(timer);
        uploadProgress.classList.remove('active');
        showToast(err.message || 'Upload failed', 'error');
    }
}

// Documents
async function loadDocuments() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        if (!res.ok) throw new Error('Failed to load documents');
        const docs = await res.json();
        renderDocuments(docs);
    } catch (err) {
        console.error(err);
    }
}

function renderDocuments(docs) {
    docCount = docs.length;
    docCountBadge.textContent = String(docCount);
    setStatus();
    if (!docs.length) {
        documentsList.innerHTML = '<div class="empty-docs">No documents uploaded yet. Upload a PDF to begin.</div>';
        return;
    }
    documentsList.innerHTML = docs.map(d => `
        <div class="doc-item">
            <div class="doc-row">
                <div class="doc-name">📄 ${escapeHtml(d.name)}</div>
                <button class="doc-del" title="Remove" onclick="deleteDocument('${escapeHtml(d.id)}')">✕</button>
            </div>
            <div class="doc-meta">${d.pages} pages · ${d.chunks} sections</div>
        </div>
    `).join('');
}

async function deleteDocument(id) {
    if (!confirm('Remove this document?')) return;
    try {
        const res = await fetch(`${API_BASE}/documents/${id}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || 'Delete failed');
        }
        showToast('Document removed', 'success');
        await loadDocuments();
    } catch (err) {
        showToast(err.message || 'Failed to remove document', 'error');
    }
}

// Chat
sendBtn.addEventListener('click', submitChat);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitChat(); }
});
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
});

clearChatBtn.addEventListener('click', () => {
    conversationId = null;
    chatMessages.innerHTML = heroHtml();
    bindSuggestions();
    showToast('Started a new conversation', 'info');
});

function heroHtml() {
    return `
        <div class="hero" id="hero">
            <h3>Welcome to DocChat</h3>
            <p>Upload a PDF (report, guideline, paper, or book) and ask questions. Answers include citations from your uploaded documents.</p>
            <div class="suggestions" id="suggestions">
                <button class="chip" data-q="Summarize the key points of this document.">Summarize key points</button>
                <button class="chip" data-q="What are the main findings or conclusions?">Main findings</button>
                <button class="chip" data-q="Are there any important recommendations or warnings?">Recommendations</button>
                <button class="chip" data-q="Explain the most important terms used in this document.">Explain key terms</button>
            </div>
        </div>`;
}

function bindSuggestions() {
    document.querySelectorAll('.chip[data-q]').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.dataset.q || '';
            chatInput.dispatchEvent(new Event('input'));
            submitChat();
        });
    });
}

function submitChat() {
    const text = chatInput.value.trim();
    if (!text || isWaiting) return;
    if (docCount === 0) {
        showToast('Please upload a PDF document first.', 'warning');
        openSidebar();
        return;
    }
    sendMessage(text);
}

async function sendMessage(question) {
    document.getElementById('hero')?.remove();
    appendMessage('user', question);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    isWaiting = true; sendBtn.disabled = true; chatInput.disabled = true;
    showTyping();

    try {
        const body = { question };
        if (conversationId) body.conversation_id = conversationId;
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
        });
        hideTyping();
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed (${res.status})`);
        }
        const data = await res.json();
        conversationId = data.conversation_id;
        appendMessage('assistant', data.answer || '', data.sources || [], data.has_answer, data.debug);
    } catch (err) {
        hideTyping();
        appendMessage('assistant', 'Sorry, something went wrong while answering. Please try again.', [], false);
        showToast(err.message || 'Failed to get response', 'error');
    } finally {
        isWaiting = false; sendBtn.disabled = false; chatInput.disabled = false; chatInput.focus();
    }
}

function appendMessage(role, text, sources = [], hasAnswer = true, debug = null) {
    const wrap = document.createElement('div');
    wrap.className = `message ${role}` + (role === 'assistant' && hasAnswer === false ? ' no-answer' : '');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (role === 'assistant') {
        bubble.innerHTML = formatAssistantText(text);
    } else {
        bubble.textContent = text;
    }
    wrap.appendChild(bubble);

    if (role === 'assistant' && (sources.length || debug)) {
        const srcWrap = document.createElement('div');
        srcWrap.className = 'sources';
        const btn = document.createElement('button');
        btn.className = 'sources-toggle';
        const toggleText = sources.length
            ? `Show ${sources.length} source${sources.length > 1 ? 's' : ''} + retrieval details`
            : 'Show retrieval details';
        btn.textContent = toggleText;
        const list = document.createElement('div');
        list.className = 'sources-list';

        const debugHtml = debug ? `
            <div class="debug-item">
                <div class="debug-title">Retrieval debug</div>
                <div class="debug-line"><strong>Original:</strong> ${escapeHtml(debug.original_question || '')}</div>
                <div class="debug-line"><strong>Rewritten:</strong> ${escapeHtml(debug.rewritten_question || '')}</div>
                <div class="debug-line"><strong>Chunks:</strong></div>
                ${(debug.chunks || []).map((c, i) => `
                    <div class="debug-chunk">
                        <span class="debug-chunk-head">#${i + 1} · ${escapeHtml(c.document)} · Page ${c.page} · score ${c.score}</span>
                        <span class="debug-chunk-text">${escapeHtml(c.text || '')}</span>
                    </div>`).join('')}
            </div>` : '';

        list.innerHTML = debugHtml + sources.map(s => `
            <div class="source-item">
                <div class="source-title">${escapeHtml(s.document)} — Page ${s.page}</div>
                <div class="source-text">"${escapeHtml((s.text || '').slice(0, 220))}${(s.text||'').length>220?'…':''}"</div>
            </div>`).join('');
        btn.addEventListener('click', () => {
            list.classList.toggle('open');
            btn.textContent = list.classList.contains('open') ? 'Hide sources' : toggleText;
        });
        srcWrap.append(btn, list);
        wrap.appendChild(srcWrap);
    }

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    wrap.appendChild(meta);

    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    const el = document.createElement('div');
    el.className = 'typing'; el.id = 'typing';
    el.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideTyping() { document.getElementById('typing')?.remove(); }

document.addEventListener('DOMContentLoaded', () => {
    bindSuggestions();
    loadDocuments();
    chatInput.focus();
});