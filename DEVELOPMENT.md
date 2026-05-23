# Development Guide

This guide provides setup instructions for local development of the Love Music Platform.

## Prerequisites

- Java 17+
- Maven 3.9+
- MySQL 8.0+
- Node.js 16+
- Docker & Docker Compose (optional but recommended)
- WeChat Developer Tools (for mini program development)
- Git

## Backend Development

### Setup Local MySQL

#### Option 1: Using Docker (Recommended)

```bash
docker-compose up mysql -d
```

#### Option 2: Manual Installation

```bash
# Install MySQL
sudo apt install mysql-server -y

# Create database and user
mysql -u root -p << EOF
CREATE DATABASE love_music;
CREATE USER 'love_user'@'localhost' IDENTIFIED BY 'love_password';
GRANT ALL PRIVILEGES ON love_music.* TO 'love_user'@'localhost';
FLUSH PRIVILEGES;
EOF
```

### Build and Run Backend

```bash
cd backend

# Build project
mvn clean package

# Run locally
mvn spring-boot:run

# Or run JAR directly
java -jar target/music-api-1.0.0.jar
```

API will be available at `http://localhost:8080/api`

### IDE Setup (IntelliJ IDEA)

1. Open `backend` folder as project
2. Install Lombok plugin: **Preferences** → **Plugins** → Search "Lombok"
3. Enable annotation processing: **Preferences** → **Build, Execution, Deployment** → **Compiler** → **Annotation Processors** → Enable
4. Run → **Edit Configurations** → Add Spring Boot configuration
5. Select `com.love.MusicApiApplication` as main class

### IDE Setup (VS Code)

1. Install Extension Pack for Java
2. Install Lombok Annotations Support for VS Code
3. Open folder in VS Code
4. Spring Boot will auto-detect and offer to create run configuration

### Common Maven Commands

```bash
# Clean and build
mvn clean package

# Run tests
mvn test

# Run specific test class
mvn test -Dtest=UserRepositoryTest

# Skip tests during build
mvn clean package -DskipTests

# Generate Javadoc
mvn javadoc:javadoc

# Check dependencies
mvn dependency:tree
```

### Database Migrations

Flyway automatically handles migrations on startup. To manually run:

```bash
# View migration status
mvn flyway:info

# Migrate database
mvn flyway:migrate

# Repair broken migration
mvn flyway:repair
```

Migrations are in `src/main/resources/db/migration/`

### Adding New Entities

1. Create entity class in `src/main/java/com/love/entity/`
2. Add corresponding repository in `src/main/java/com/love/repository/`
3. Add service in `src/main/java/com/love/service/`
4. Add controller in `src/main/java/com/love/controller/`
5. Create database migration in `src/main/resources/db/migration/`

Example:

```java
// Entity
@Entity
@Table(name = "my_entity")
public class MyEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    // ... fields
}

// Repository
@Repository
public interface MyEntityRepository extends JpaRepository<MyEntity, Long> {
}

// Service
@Service
public class MyEntityService {
    // ... business logic
}

// Controller
@RestController
@RequestMapping("/my-entity")
public class MyEntityController {
    // ... endpoints
}
```

### Debugging

```bash
# Enable debug logging
export SPRING_PROFILES_ACTIVE=debug
mvn spring-boot:run

# Or in application.yml
logging:
  level:
    com.love: DEBUG
    org.springframework: DEBUG
```

## Mini Program Development

### Setup WeChat Developer Tools

1. Download from: https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html
2. Install and open
3. Scan QR code to login
4. Click "Open" and select `miniprogram` folder from this project

### Project Structure

```
miniprogram/
├── pages/           # Page components (each with .js, .wxml, .wxss, .json)
├── utils/           # Utility functions
├── images/          # Image assets
├── app.js           # App configuration and lifecycle
├── app.json         # Global configuration
├── app.wxss         # Global styles
└── sitemap.json     # SEO configuration
```

### Page Structure

Each page has 4 files:

```
pages/mypage/
├── mypage.js        # Page logic
├── mypage.wxml      # Template (like HTML)
├── mypage.wxss      # Styles (like CSS)
└── mypage.json      # Page configuration (optional)
```

### Hot Reload

In WeChat Developer Tools:
- Enable "Hot Reload on File Save" in settings
- Compile button compiles after save
- Use "Compile and Preview" to test

### Testing on Device

1. Click "Preview" in WeChat Developer Tools
2. Scan QR code with WeChat (not WeChat Developer)
3. App opens in WeChat for testing
4. Console errors visible in developer tools

### Common wxapi Functions

```javascript
// HTTP request
wx.request({
  url: 'https://api.example.com/data',
  method: 'GET',
  success: (res) => { },
  fail: (err) => { }
})

// Show toast
wx.showToast({
  title: 'Success',
  icon: 'success',
  duration: 1500
})

// Show loading
wx.showLoading({ title: 'Loading...' })
wx.hideLoading()

// Navigate
wx.navigateTo({ url: '/pages/index/index' })
wx.navigateBack({ delta: 1 })

// Storage
wx.setStorageSync('key', value)
wx.getStorageSync('key')
wx.removeStorageSync('key')

// Audio playback
const bgAudioManager = wx.getBackgroundAudioManager()
bgAudioManager.src = 'http://example.com/audio.mp3'
bgAudioManager.play()
```

### WeChat Login Flow for Testing

```javascript
// In your code
wx.login({
  success: (res) => {
    const code = res.code
    // Send code to backend
    app.request({
      url: '/auth/wx-login',
      method: 'POST',
      data: { code }
    })
  }
})
```

Note: WeChat `code` is only valid for testing in developer tools. Production requires real WeChat.

### Debugging Mini Program

1. **Enable Debugging in Developer Tools**
   - Open Developer Tools
   - Click "Console" tab
   - View real-time logs and errors

2. **Add Debug Statements**
   ```javascript
   console.log('Debug message:', data)
   ```

3. **Use Debugger**
   - Click "Sources" tab
   - Set breakpoints
   - Step through code

4. **Inspect DOM**
   - Click "Elements" tab
   - Inspect page structure

## API Development Workflow

### Test API Endpoints

Using curl:
```bash
# Login
curl -X POST http://localhost:8080/api/auth/wx-login \
  -H "Content-Type: application/json" \
  -d '{"code":"test_code"}'

# List music
curl -X GET "http://localhost:8080/api/music/list?page=1&size=20"

# Get with JWT token
curl -X GET http://localhost:8080/api/auth/me \
  -H "Authorization: Bearer <token>"
```

Using Postman:
1. Download and install Postman
2. Create collection for API endpoints
3. Set variables for baseUrl and token
4. Test endpoints

### Adding New Endpoint

1. Add method to Controller:
```java
@GetMapping("/{id}")
public ResponseEntity<ApiResponse<?>> getItem(@PathVariable Long id) {
    // Implementation
}
```

2. Add service method:
```java
public Item getItem(Long id) {
    return repository.findById(id).orElse(null);
}
```

3. Update security config if needed:
```java
.requestMatchers("/api/public/**").permitAll()
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add my feature"

# Push to remote
git push origin feature/my-feature

# Create pull request on GitHub
# After review and approval, merge to main
```

## Common Issues and Solutions

### Maven Build Fails

```bash
# Clear cache
rm -rf ~/.m2/repository
mvn clean package

# Update dependencies
mvn dependency:resolve
```

### Spring Boot won't start

```bash
# Check logs
tail -f logs/app.log

# Check port availability
lsof -i :8080

# Verify database connection
mysql -u love_user -ppassword -h localhost love_music
```

### Mini Program networking issues

```bash
# Check API baseUrl
console.log(app.globalData.apiBaseUrl)

# Verify domain in WeChat console
# Check CORS headers in backend

# Test with curl from server
curl -X GET http://localhost:8080/api/music/list
```

### Hot reload not working

- Check file save detected
- Clear cache: Ctrl+Shift+R
- Restart developer tools
- Check file permissions

## Performance Tips

### Backend
- Use pagination for large datasets
- Add database indexes
- Cache frequently accessed data with Redis
- Use lazy loading for relationships
- Monitor slow queries

### Frontend
- Minimize API calls
- Cache images locally
- Lazy load images
- Batch operations where possible
- Monitor network tab in dev tools

## Code Style

### Backend (Java)
- Follow Google Java Style Guide
- Use meaningful variable names
- Add Javadoc for public methods
- Keep methods focused and small
- Use final for constants

### Frontend (JavaScript)
- Use camelCase for variables
- Use UPPER_SNAKE_CASE for constants
- Add comments for complex logic
- Use async/await over callbacks
- Handle errors properly

## Documentation

Update documentation when:
- Adding new API endpoints
- Changing database schema
- Modifying configuration
- Adding new features
- Fixing bugs

---

Happy coding! 🚀
