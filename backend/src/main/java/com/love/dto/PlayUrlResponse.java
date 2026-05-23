package com.love.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlayUrlResponse {
    private String playUrl;
    private Long expiresIn;  // seconds
    private Long musicId;
}
