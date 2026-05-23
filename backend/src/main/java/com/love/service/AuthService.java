package com.love.service;

import com.google.gson.Gson;
import com.love.dto.LoginResponse;
import com.love.entity.User;
import com.love.repository.UserRepository;
import com.love.security.JwtProvider;
import lombok.extern.slf4j.Slf4j;
import org.apache.hc.client5.http.classic.HttpClient;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ClassicHttpResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Map;

@Slf4j
@Service
public class AuthService {
    @Value("${wechat.appid}")
    private String wechatAppId;

    @Value("${wechat.appsecret}")
    private String wechatAppSecret;

    @Value("${wechat.auth-url}")
    private String wechatAuthUrl;

    @Value("${jwt.expiration}")
    private long jwtExpiration;

    private final UserRepository userRepository;
    private final JwtProvider jwtProvider;
    private final Gson gson = new Gson();

    public AuthService(UserRepository userRepository, JwtProvider jwtProvider) {
        this.userRepository = userRepository;
        this.jwtProvider = jwtProvider;
    }

    public LoginResponse wechatLogin(String code) {
        try {
            // Exchange code for openid and session_key
            String url = String.format("%s?appid=%s&secret=%s&js_code=%s&grant_type=authorization_code",
                    wechatAuthUrl, wechatAppId, wechatAppSecret, code);

            HttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);

            try (ClassicHttpResponse response = (ClassicHttpResponse) httpClient.execute(httpGet)) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(response.getEntity().getContent()));
                StringBuilder result = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    result.append(line);
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> data = gson.fromJson(result.toString(), Map.class);

                if (data.containsKey("errcode")) {
                    throw new RuntimeException("WeChat auth failed: " + data.get("errmsg"));
                }

                String openid = (String) data.get("openid");
                String sessionKey = (String) data.get("session_key");

                // Find or create user
                User user = userRepository.findByOpenid(openid).orElse(null);
                if (user == null) {
                    user = new User();
                    user.setOpenid(openid);
                    user.setNickname("User_" + openid.substring(0, 8));
                    user.setStatus((byte) 1);
                    user = userRepository.save(user);
                }

                // Generate JWT token
                String token = jwtProvider.generateToken(user.getId());

                return new LoginResponse(
                        token,
                        user.getId(),
                        user.getNickname(),
                        user.getAvatarUrl(),
                        jwtExpiration / 1000  // Convert to seconds
                );
            }
        } catch (Exception e) {
            log.error("WeChat login failed", e);
            throw new RuntimeException("WeChat login failed: " + e.getMessage());
        }
    }

    public User getCurrentUser(Long userId) {
        return userRepository.findById(userId).orElse(null);
    }
}
