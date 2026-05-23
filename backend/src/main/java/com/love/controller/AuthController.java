package com.love.controller;

import com.love.dto.ApiResponse;
import com.love.dto.LoginResponse;
import com.love.dto.WeChatLoginRequest;
import com.love.service.AuthService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/auth")
public class AuthController {
    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/wx-login")
    public ResponseEntity<ApiResponse<LoginResponse>> wechatLogin(@RequestBody WeChatLoginRequest request) {
        try {
            LoginResponse response = authService.wechatLogin(request.getCode());
            return ResponseEntity.ok(ApiResponse.success(response));
        } catch (Exception e) {
            log.error("Login failed", e);
            return ResponseEntity.status(401)
                    .body(ApiResponse.error(401, e.getMessage()));
        }
    }

    @GetMapping("/me")
    public ResponseEntity<ApiResponse<?>> getCurrentUser() {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            var user = authService.getCurrentUser(userId);
            return ResponseEntity.ok(ApiResponse.success(user));
        } catch (Exception e) {
            log.error("Get current user failed", e);
            return ResponseEntity.status(401)
                    .body(ApiResponse.error(401, "Unauthorized"));
        }
    }
}
