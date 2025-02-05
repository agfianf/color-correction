# report_generator.py
from datetime import datetime
from importlib import resources

import pandas as pd

from color_correction.utils.formater import format_value


class ReportGenerator:
    def __init__(self) -> None:
        self.template_dir = "color_correction.templates"

    def _read_template(self, filename: str) -> str:
        """Read template file content"""
        with resources.files(self.template_dir).joinpath(filename).open("r") as f:
            return f.read()

    def generate_report(self, body_report: str) -> str:
        """Generate full HTML report"""
        # Load components
        styles = self._read_template("style-report.css")
        scripts = self._read_template("script-report.js")
        column_desc = self._read_template("column_description.html")
        base_template = self._read_template("base_report.html")

        current_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")

        # Replace placeholders
        final_html = base_template.format(
            styles=styles,
            scripts=scripts,
            current_time=current_time,
            body_report=body_report,
            column_descriptions=column_desc,
        )
        return final_html

    def generate_table(self, headers: list, rows: list) -> str:
        """Generate table HTML"""
        headers_html = "".join([f"<th>{h}</th>" for h in headers])
        rows_html = "".join([f"<tr>{r}</tr>" for r in rows])

        return f"""
        <table>
            <thead>
                <tr>{headers_html}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """

    def generate_html_report(self, df: pd.DataFrame, path_html: str) -> str:
        """Generate HTML report from DataFrame with images"""
        df_html = df.copy()

        # Generate rows
        rows = []
        for _, row in df_html.iterrows():
            row_html = "<tr>"
            for value in row:
                formatted_value = format_value(value)
                row_html += f"<td>{formatted_value}</td>"
            row_html += "</tr>"
            rows.append(row_html)

        table_html = self.generate_table(
            headers=df.columns.tolist(),
            rows=rows,
        )
        report_html = self.generate_report(table_html)

        with open(path_html, "w") as f:
            f.write(report_html)

        return report_html

    def save_dataframe(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame with pickle to preserve numpy arrays"""
        df.to_pickle(filepath)
