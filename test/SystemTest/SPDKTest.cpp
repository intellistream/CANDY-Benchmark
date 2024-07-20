//
// Created by tony on 19/07/24.
//
#include <vector>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include <Utils/ConfigMap.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <spdk/env.h>
#include <spdk/nvme.h>
#include <spdk/stdinc.h>
#define NUM_THREADS 4
#define BASE_LBA 0
#define LBA_INCREMENT 1
static struct spdk_nvme_ctrlr *g_ctrlr = NULL;
static struct spdk_nvme_ns *g_ns = NULL;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_context {
  struct spdk_nvme_qpair *qpair;
  uint32_t lba;
};

static void read_complete(void *arg, const struct spdk_nvme_cpl *completion);
static void write_complete(void *arg, const struct spdk_nvme_cpl *completion);
static int initialize_spdk(void);
static void cleanup_spdk(void);
static void *thread_func(void *arg);

int tmain() {
  if (initialize_spdk() != 0) {
    fprintf(stderr, "Failed to initialize SPDK\n");
    return -1;
  }

  pthread_t threads[NUM_THREADS];
  struct thread_context contexts[NUM_THREADS];
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(g_ns);
  printf("Sector size=%d",sector_size);
  for (int i = 0; i < NUM_THREADS; i++) {
    contexts[i].lba = BASE_LBA + i * LBA_INCREMENT;
    contexts[i].qpair = spdk_nvme_ctrlr_alloc_io_qpair(g_ctrlr, NULL, 0);
    if (contexts[i].qpair == NULL) {
      fprintf(stderr, "Failed to allocate I/O queue pair for thread %d\n", i);
      cleanup_spdk();
      return -1;
    }
    pthread_create(&threads[i], NULL, thread_func, &contexts[i]);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
    spdk_nvme_ctrlr_free_io_qpair(contexts[i].qpair);
  }

  cleanup_spdk();
  return 0;
}

static void *thread_func(void *arg) {
  struct thread_context *ctx = (struct thread_context *)arg;
  uint32_t sector_size = spdk_nvme_ns_get_sector_size(g_ns);
  uint32_t *buffer = (uint32_t *)spdk_zmalloc(sector_size, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  if (buffer == NULL) {
    fprintf(stderr, "Failed to allocate buffer\n");
    pthread_exit(NULL);
  }

  // Read the integer from the SSD
  printf("Thread %lu: Reading integer from SSD at LBA %u...\n", pthread_self(), ctx->lba);
  if (spdk_nvme_ns_cmd_read(g_ns, ctx->qpair, buffer, ctx->lba, 1, read_complete, NULL, 0) != 0) {
    fprintf(stderr, "Thread %lu: Read command failed\n", pthread_self());
    spdk_free(buffer);
    pthread_exit(NULL);
  }
  while (!spdk_nvme_qpair_process_completions(ctx->qpair, 0));

  // Increment the integer
  buffer[0] += 1;

  // Write the incremented integer back to the SSD
  printf("Thread %lu: Writing incremented integer back to SSD at LBA %u...\n", pthread_self(), ctx->lba);
  if (spdk_nvme_ns_cmd_write(g_ns, ctx->qpair, buffer, ctx->lba, 1, write_complete, NULL, 0) != 0) {
    fprintf(stderr, "Thread %lu: Write command failed\n", pthread_self());
    spdk_free(buffer);
    pthread_exit(NULL);
  }
  while (!spdk_nvme_qpair_process_completions(ctx->qpair, 0));

  // Read the integer again to verify
  printf("Thread %lu: Reading incremented integer from SSD at LBA %u...\n", pthread_self(), ctx->lba);
  if (spdk_nvme_ns_cmd_read(g_ns, ctx->qpair, buffer, ctx->lba, 1, read_complete, NULL, 0) != 0) {
    fprintf(stderr, "Thread %lu: Read command failed\n", pthread_self());
    spdk_free(buffer);
    pthread_exit(NULL);
  }
  while (!spdk_nvme_qpair_process_completions(ctx->qpair, 0));

  // Print the incremented integer
  printf("Thread %lu: Incremented integer at LBA %u: %u\n", pthread_self(), ctx->lba, buffer[0]);

  spdk_free(buffer);
  pthread_exit(NULL);
}

static void read_complete(void *arg, const struct spdk_nvme_cpl *completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    fprintf(stderr, "Read operation failed\n");
    exit(-1);
  }
}

static void write_complete(void *arg, const struct spdk_nvme_cpl *completion) {
  if (spdk_nvme_cpl_is_error(completion)) {
    fprintf(stderr, "Write operation failed\n");
    exit(-1);
  }
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts) {
  printf("Attaching to %s\n", trid->traddr);
  return true;
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
  g_ctrlr = ctrlr;
}

static int initialize_spdk(void) {
  struct spdk_env_opts opts;
  spdk_env_opts_init(&opts);
  opts.name = "spdk_example";
  if (spdk_env_init(&opts) < 0) {
    fprintf(stderr, "Unable to initialize SPDK environment\n");
    return -1;
  }

  if (spdk_nvme_probe(NULL, NULL, probe_cb, attach_cb, NULL) != 0) {
    fprintf(stderr, "spdk_nvme_probe() failed\n");
    spdk_env_fini();
    return -1;
  }

  if (g_ctrlr == NULL) {
    fprintf(stderr, "No NVMe controllers found\n");
    spdk_env_fini();
    return -1;
  }

  g_ns = spdk_nvme_ctrlr_get_ns(g_ctrlr, spdk_nvme_ctrlr_get_first_active_ns(g_ctrlr));
  if (g_ns == NULL) {
    fprintf(stderr, "Namespace not found\n");
    spdk_nvme_detach(g_ctrlr);
    spdk_env_fini();
    return -1;
  }

  return 0;
}

static void cleanup_spdk(void) {
  if (g_ctrlr != NULL) {
    spdk_nvme_detach(g_ctrlr);
  }
  spdk_env_fini();
}

using namespace std;

using namespace std;

TEST_CASE("Test SPDK", "[short]")
{
  int a = 0;
  // place your test here
  tmain();
  REQUIRE(a == 0);
}
