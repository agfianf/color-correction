document.addEventListener('DOMContentLoaded', function() {
    const getCellValue = (tr, idx) => {
        // Pastikan tr dan children[idx] ada
        if (!tr || !tr.children || !tr.children[idx]) {
            return '';
        }

        const cell = tr.children[idx];

        // Skip jika cell kosong
        if (!cell) {
            return '';
        }

        // Check if cell contains an image
        if (cell.querySelector('img')) {
            return ''; // Skip sorting for images
        }

        // Get text content, trim whitespace
        const content = cell.textContent.trim();

        // If empty content, return empty string
        if (!content) {
            return '';
        }

        // Try parse as number first
        const num = Number(content);
        if (!isNaN(num)) {
            return num;
        }

        // Try parse as JSON
        try {
            return JSON.parse(content);
        } catch {
            return content;
        }
    };

    const comparer = (idx) => (a, b) => {
        const v1 = getCellValue(a, idx);
        const v2 = getCellValue(b, idx);

        // Handle empty values
        if (v1 === '' && v2 === '') return 0;
        if (v1 === '') return 1;
        if (v2 === '') return -1;

        // Handle different types
        if (typeof v1 === 'number' && typeof v2 === 'number') {
            return v1 - v2;
        }

        if (typeof v1 === 'object' && typeof v2 === 'object') {
            return JSON.stringify(v1).localeCompare(JSON.stringify(v2));
        }

        return v1.toString().localeCompare(v2.toString());
    };

    document.querySelectorAll('table thead th').forEach((th, index) => {
        let asc = true; // Initial sort direction

        th.addEventListener('click', function() {
            const tbody = this.closest('table').querySelector('tbody');
            if (!tbody) return;

            const rows = Array.from(tbody.querySelectorAll('tr'));
            if (!rows.length) return;

            // Sort rows
            rows.sort((a, b) => {
                const result = comparer(index)(a, b);
                return asc ? result : -result;
            });

            // Clear existing rows
            while (tbody.firstChild) {
                tbody.removeChild(tbody.firstChild);
            }

            // Add sorted rows
            tbody.append(...rows);

            // Update sort direction indicators
            th.closest('tr').querySelectorAll('th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            th.classList.add(asc ? 'sort-asc' : 'sort-desc');

            // Toggle direction for next click
            asc = !asc;
        });
    });
});

function exportToPDF() {
    const element = document.querySelector('.report-container');
    const opt = {
        margin: 1,
        filename: 'color-correction-report.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'a4', orientation: 'landscape' }
    };

    const btn = document.querySelector('.export-btn');
    btn.textContent = 'Generating PDF...';
    btn.disabled = true;

    html2pdf().set(opt).from(element).save()
        .then(() => {
            btn.textContent = 'Export to PDF';
            btn.disabled = false;
        });
}
