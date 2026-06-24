import asyncio

async def do_some_processing():
    print("Task 1 Started")
    await asyncio.sleep(3)
    print("Task 1 Completed")
    return "Done"

async def do_other_processing():
    print("Task 2 Started")
    await asyncio.sleep(5)
    print("Task 2 Completed")
    return "Done 2"

async def do_yet_more_processing():
    print("Task 3 Started")
    await asyncio.sleep(7)
    print("Task 3 Completed")
    return "Done 3"

async def main():
    print("Starting gather...")

    results = await asyncio.gather(
        do_some_processing(),
        do_other_processing(),
        do_yet_more_processing()
    )

    print(results)

asyncio.run(main())