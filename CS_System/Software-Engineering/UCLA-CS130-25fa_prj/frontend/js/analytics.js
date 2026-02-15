// note: written with GenAI assistance
document.addEventListener('alpine:init', () => {
    Alpine.data('analyticsDashboard', () => ({
        topUrls: [],
        searchCode: '',
        linkStats: null,
        error: '',
        
        init() {
            this.loadTopUrls();
        },

        async loadTopUrls() {
            try {
                const res = await fetch('/analytics/top/10');
                if (res.ok) {
                    this.topUrls = await res.json();
                }
            } catch (e) {
                console.error('Failed to load top URLs', e);
            }
        },

        async loadLinkStats() {
            if (!this.searchCode) return;
            this.error = '';
            this.linkStats = null;

            try {
                const res = await fetch(`/analytics/${this.searchCode}`);
                if (res.status === 404) throw new Error('Short code not found');
                if (!res.ok) throw new Error('Failed to load stats');

                this.linkStats = await res.json();
            } catch (err) {
                this.error = err.message;
            }
        }
    }));
});

