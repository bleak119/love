package com.love.repository;

import com.love.entity.PlayHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PlayHistoryRepository extends JpaRepository<PlayHistory, Long> {
    Page<PlayHistory> findByUserIdOrderByPlayedAtDesc(Long userId, Pageable pageable);
}
