from apscheduler.schedulers.blocking import BlockingScheduler
from rich.console import Console
from rich.table import Table
import pandas as pd
from bay_area_scraper import BayAreaScraper
import smtplib
from email.message import EmailMessage
import os
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


console = Console()

class RealTimeMonitor:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.scraper = BayAreaScraper()
        
    def _update_dashboard(self):
        """Rich terminal dashboard"""
        try:
            df = pd.read_csv('data/bay_area_startups.csv')
            table = Table(title="Bay Area Startup Monitor", show_lines=True)
            
            table.add_column("Name", style="cyan")
            table.add_column("Funding", style="green")
            table.add_column("Source", style="magenta")
            table.add_column("Last Updated", style="yellow")
            
            for _, row in df.tail(5).iterrows():
                table.add_row(
                    row['name'],
                    f"${row['funding']/1e6:.1f}M" if pd.notnull(row['funding']) else "Undisclosed",
                    row['source'],
                    pd.to_datetime(row['timestamp']).strftime('%m/%d %H:%M')
                )
                
            console.clear()
            console.print(table)
        except FileNotFoundError:
            console.print("No data file found", style="red")

    def _run_scraper(self):
        """Scheduled scraping job"""
        self.scraper.scrape_growthlist()
        self.scraper.scrape_ycombinator()
        self.scraper.scrape_crunchbase()
        self.scraper.clean_data()
        self.scraper.save_data()
        self._update_dashboard()

    def start(self):
        """Start monitoring system"""
        self.scheduler.add_job(self._run_scraper, 'interval', hours=6)
        console.print("🚀 Starting real-time monitoring...", style="bold green")
        self.scheduler.start()






class AlertSystem:
    def __init__(self):
        self.email = os.getenv('EMAIL')
        self.password = os.getenv('APP_PASSWORD')
        self.recipient = os.getenv('RECIPIENT')
        
    def send_alert(self, message):
        msg = EmailMessage()
        msg.set_content(f"Bay Area Startup Alert:\n\n{message}")
        msg['Subject'] = 'Startup Monitoring Alert'
        msg['From'] = self.email
        msg['To'] = self.recipient
        
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            logger.success("Alert email sent")
        except Exception as e:
            logger.error(f"Email failed: {str(e)}")

if __name__ == "__main__":
    monitor = RealTimeMonitor()
    monitor.start()
