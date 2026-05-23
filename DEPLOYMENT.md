# Deployment Guide for Love Music Platform

This guide provides step-by-step instructions for deploying the Love Music Platform to a production environment.

## Prerequisites

- Cloud server with Ubuntu 20.04+ LTS
- Domain name with DNS configured
- WeChat Developer Account with Mini Program registered
- Docker and Docker Compose installed

## Environment Setup

### 1. Install Docker and Docker Compose

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/bleak119/love.git
sudo chown -R $USER:$USER love
cd love
```

### 3. Configure Environment

```bash
# Create .env file
cat > .env << EOF
WECHAT_APPID=your_actual_appid_here
WECHAT_APPSECRET=your_actual_appsecret_here
JWT_SECRET=$(openssl rand -base64 32)
MYSQL_ROOT_PASSWORD=$(openssl rand -base64 16)
MYSQL_PASSWORD=$(openssl rand -base64 16)
EOF

chmod 600 .env
```

### 4. Configure SSL Certificate

```bash
# Create SSL directory
mkdir -p ssl

# Option A: Using Let's Encrypt (Recommended)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot certonly --standalone -d your-domain.com -d www.your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
sudo chown $USER:$USER ssl/*

# Option B: Using self-signed certificate (for testing only)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
```

### 5. Update Configuration Files

Edit `nginx.conf`:
```bash
# Uncomment HTTPS section
# Update server_name to your domain
# Uncomment SSL certificate paths
```

Update `docker-compose.yml`:
```bash
# Change api.xxx.com to your domain
# Update MySQL passwords (already in .env)
```

Update `backend/src/main/resources/application.yml`:
```bash
# Change datasource URL to production database
# Update WeChat configuration
```

## Deployment Steps

### 1. Build and Start Services

```bash
# Create data directories
mkdir -p /data/music logs

# Build backend image
docker-compose build

# Start all services in background
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Verify Services

```bash
# Check MySQL
docker-compose exec mysql mysql -u root -ppassword -e "SELECT VERSION();"

# Check API
curl -s http://localhost:8080/api/health | jq .

# Check Nginx
curl -s http://localhost/health
```

### 3. Database Initialization

The database is automatically initialized via Flyway migrations on first startup. To verify:

```bash
docker-compose exec mysql mysql -u love_user -ppassword love_music -e "SHOW TABLES;"
```

### 4. Configure WeChat Mini Program

1. Go to [WeChat Mini Program Console](https://mp.weixin.qq.com)
2. Navigate to **Settings** → **Server Domain Configuration**
3. Add your domain to:
   - Request合法域名: `https://your-domain.com`
   - Download文件地址: `https://your-domain.com`
   - Upload文件地址: `https://your-domain.com`

4. Update `miniprogram/app.js`:
```javascript
globalData: {
  apiBaseUrl: 'https://your-domain.com/api'
}
```

## Maintenance

### Regular Backups

```bash
# Backup database daily
cat > /opt/love/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/love/backups"
mkdir -p $BACKUP_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T mysql mysqldump -u love_user -ppassword love_music | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz
# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
EOF

chmod +x /opt/love/backup.sh

# Add to crontab
crontab -e
# Add: 2 3 * * * /opt/love/backup.sh
```

### SSL Certificate Renewal

```bash
# Renew Let's Encrypt certificate
sudo certbot renew --quiet

# Restart Nginx if renewed
docker-compose restart nginx
```

### Monitoring and Logs

```bash
# View API logs
docker-compose logs api -f

# View MySQL logs
docker-compose logs mysql -f

# View Nginx logs
docker-compose logs nginx -f

# Check disk usage
du -sh /data/music /opt/love
```

### Scaling

If needed, scale services:

```bash
# Scale API servers (requires load balancing setup)
docker-compose up -d --scale api=3

# For production, use Kubernetes or other orchestration
```

## Troubleshooting

### Services won't start

```bash
# Check Docker daemon
sudo systemctl status docker

# Check compose file syntax
docker-compose config

# Check resource constraints
docker stats
```

### Database connection errors

```bash
# Check MySQL container
docker-compose exec mysql mysql -u root -ppassword -e "SHOW PROCESSLIST;"

# Check database exists
docker-compose exec mysql mysql -u root -ppassword -e "SHOW DATABASES;"
```

### WeChat login fails

```bash
# Check API connectivity
curl -X POST http://localhost:8080/api/auth/wx-login \
  -H "Content-Type: application/json" \
  -d '{"code":"test"}'

# Check firewall rules
sudo ufw status
```

### Music playback issues

```bash
# Check file permissions
ls -la /data/music/

# Verify nginx can read files
docker-compose exec nginx ls -la /usr/share/nginx/html/music/
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Install UFW
sudo apt install ufw -y

# Enable firewall
sudo ufw enable

# Allow SSH
sudo ufw allow 22

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Check rules
sudo ufw status
```

### 2. Fail2Ban Setup

```bash
# Install Fail2Ban
sudo apt install fail2ban -y

# Create local config
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Enable and start
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 3. Regular Updates

```bash
# Set up automatic security updates
sudo apt install unattended-upgrades -y

# Enable automatic updates
sudo dpkg-reconfigure unattended-upgrades
```

## Performance Tuning

### MySQL Optimization

```bash
# Adjust MySQL configuration in docker-compose.yml
# Add custom my.cnf volume

# Monitor slow queries
docker-compose exec mysql mysql -u root -ppassword \
  -e "SET GLOBAL slow_query_log = 'ON';"
```

### API Performance

- Add caching headers in Nginx for static content
- Use Redis for session caching
- Enable gzip compression in Nginx
- Use CDN for music files

### Database Indexing

The migrations already create indexes on:
- user.openid
- music.status, title, artist
- favorite.user_id, music_id
- play_history.user_id, played_at

Monitor slow queries for additional optimization.

## Support and Troubleshooting

For detailed API troubleshooting, see the main README.md

For Docker issues: https://docs.docker.com/

For WeChat documentation: https://developers.weixin.qq.com/

---

Last updated: 2026-05-23
