// note: written with GenAI assistance
document.addEventListener('alpine:init', () => {
    Alpine.data('linkManager', () => ({
        // State
        createForm: { url: '', password: '' },
        manageForm: { code: '', password: '' },
        
        createdLink: null,
        managedLink: null,
        managedLinkUrl: '',
        
        isLoading: false,
        message: '',
        messageType: 'success', // 'success' or 'error'

        // Actions
        async createLink() {
            this.isLoading = true;
            this.message = '';
            this.createdLink = null;

            try {
                const payload = { url: this.createForm.url };
                if (this.createForm.password) payload.password = this.createForm.password;

                const res = await fetch('/api/link', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) throw new Error(await res.text() || 'Failed to create link');
                
                const data = await res.json();
                this.createdLink = data.code;
                this.createForm.url = '';
                this.createForm.password = '';
                this.showMsg('Short link created successfully!', 'success');
            } catch (err) {
                this.showMsg(err.message, 'error');
            } finally {
                this.isLoading = false;
            }
        },

        async loadLink() {
            if (!this.manageForm.code) return this.showMsg('Please enter a short code', 'error');
            
            this.isLoading = true;
            this.message = '';
            this.managedLink = null;

            try {
                const headers = {};
                if (this.manageForm.password) headers['Link-Password'] = this.manageForm.password;

                const res = await fetch(`/api/link/${this.manageForm.code}`, { headers });
                
                if (res.status === 403) throw new Error('Incorrect password for protected link');
                if (res.status === 404) throw new Error('Short code not found');
                if (!res.ok) throw new Error('Failed to load link');

                const data = await res.json();
                this.managedLink = data;
                this.managedLinkUrl = data.url;
            } catch (err) {
                this.showMsg(err.message, 'error');
            } finally {
                this.isLoading = false;
            }
        },

        async updateLink() {
            if (!this.managedLinkUrl) return this.showMsg('URL cannot be empty', 'error');

            this.isLoading = true;
            try {
                const headers = { 'Content-Type': 'application/json' };
                if (this.manageForm.password) headers['Link-Password'] = this.manageForm.password;

                const res = await fetch(`/api/link/${this.manageForm.code}`, {
                    method: 'PUT',
                    headers,
                    body: JSON.stringify({ url: this.managedLinkUrl })
                });

                if (res.status === 403) throw new Error('Incorrect password');
                if (!res.ok) throw new Error(await res.text() || 'Update failed');

                this.showMsg('Link updated successfully!', 'success');
                this.managedLink.url = this.managedLinkUrl; // Update local view
            } catch (err) {
                this.showMsg(err.message, 'error');
            } finally {
                this.isLoading = false;
            }
        },

        async deleteLink() {
            if (!confirm('Are you sure you want to delete this short link? This action cannot be undone.')) return;

            this.isLoading = true;
            try {
                const headers = {};
                if (this.manageForm.password) headers['Link-Password'] = this.manageForm.password;

                const res = await fetch(`/api/link/${this.manageForm.code}`, {
                    method: 'DELETE',
                    headers
                });

                if (res.status === 403) throw new Error('Incorrect password');
                if (!res.ok) throw new Error('Delete failed');

                this.showMsg('Link deleted successfully', 'success');
                this.managedLink = null;
                this.manageForm.code = '';
                this.manageForm.password = '';
            } catch (err) {
                this.showMsg(err.message, 'error');
            } finally {
                this.isLoading = false;
            }
        },

        // Helpers
        getShortUrl(code) {
            return `${window.location.origin}/l/${code}`;
        },

        copyToClipboard(text) {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text)
                    .then(() => {
                        this.showMsg('Copied to clipboard!', 'success');
                    })
                    .catch(() => {
                        this.showMsg('Failed to copy. Try Ctrl+C manually.', 'error');
                    });
            } else {
                // Fallback for older browsers or insecure contexts
                this.showMsg('Copy not supported. Please copy manually.', 'error');
            }
        },

        showMsg(text, type = 'success') {
            this.message = text;
            this.messageType = type;
            setTimeout(() => this.message = '', 5000);
        }
    }));
});

